"""Consumes queue-router's /traffic API and outputs an OpenGL visualisation
Adapted from: https://github.com/glumpy/glumpy/blob/master/examples/graph.py
"""

from collections import OrderedDict
import configparser
import json
import os
import random
import threading


from glumpy import app, collections
from glumpy.transforms import Position, OrthographicProjection, Viewport
import numpy as np
import redis
from scipy.spatial.distance import cdist

import requests


MODULE_PARENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

config = configparser.ConfigParser()
config.read(os.path.join(MODULE_PARENT_DIRECTORY, 'config.ini'))

QUEUE_ROUTER_HOST = config['QUEUE_ROUTER']['host']
QUEUE_ROUTER_TOKEN = config['QUEUE_ROUTER']['token']
TRAFFIC_API_CALL_INTERVAL = float(
    config['QUEUE_ROUTER']['traffic_api_call_interval']
)
TIME_INCREMENT_INTERVAL = float(
    config['QUEUE_ROUTER']['time_increment_interval']
)

OPEN_GL_BACKEND = config['DISPLAY']['backend']
FULL_SCREEN = config['DISPLAY']['full_screen'] == 'True'
DISPLAY_WIDTH = int(config['DISPLAY']['width'])
DISPLAY_HEIGHT = int(config['DISPLAY']['height'])
NODES_MARGIN = float(config['DISPLAY']['nodes_margin'])

MAXIMUM_NODES = int(config['NETWORK']['maximum_nodes'])
NODE_SIZE = int(config['NETWORK']['node_size'])
EDGE_WIDTH = float(config['NETWORK']['edge_width'])
ATTRACTION = float(config['NETWORK']['attraction'])
REPULSION = float(config['NETWORK']['repulsion'])
NODE_DISTANCE = float(config['NETWORK']['node_distance'])
CONNECTION_DISPLAY_CURVE_SCALE = float(
    config['NETWORK']['connection_display_curve_scale']
)
CONNECTION_DISPLAY_CURVE_ADDEND = float(
    config['NETWORK']['connection_display_curve_addend']
)
CONNECTION_DISPLAY_DURATION_SCALE = float(
    config['NETWORK']['connection_display_duration_scale']
)
BASE_CONNECTION_DISPLAY_DURATION = float(
    config['NETWORK']['base_connection_display_duration']
)


redis_client = redis.Redis()


class RouterClock:

    time = None

    def __init__(self):
        self.time = self.get_router_time()
        self.increment_time()

    @staticmethod
    def get_router_time():
        return requests.get(
            'http://{host}/time?token={token}'.format(
                host=QUEUE_ROUTER_HOST,
                token=QUEUE_ROUTER_TOKEN,
            )
        ).json()

    def increment_time(self):
        self.time += TIME_INCREMENT_INTERVAL

        threading.Timer(
            TIME_INCREMENT_INTERVAL,
            self.increment_time
        ).start()


class NodePositionManager:
    """Manages addressed node positions based on router traffic data."""

    positions = OrderedDict()

    def get_positions(self, traffic_data):
        self.prune_inactive(traffic_data)
        self.create_any_new(traffic_data)
        return np.array(list(self.positions.values()))

    def get_connections(self, traffic_data, router_time):
        connections = np.zeros((len(self.positions), len(self.positions)))

        for from_index, from_address in enumerate(self.positions.keys()):
            from_traffic = traffic_data[from_address]
            for to_index, to_address in enumerate(self.positions.keys()):
                if (
                        from_traffic and
                        to_address in from_traffic and
                        self.should_show_connection(
                            from_traffic[to_address],
                            router_time,
                        )
                ):
                    connections[from_index, to_index] = 1

        return connections

    def set_positions(self, position_array):
        for index, address in enumerate(self.positions.keys()):
            self.positions[address] = position_array[index]

    def prune_inactive(self, traffic_data):
        self.positions = {
            address: position for address, position in self.positions.items()
            if address in traffic_data
        }

    def create_any_new(self, traffic_data):
        for address in traffic_data:
            if address not in self.positions:
                self.positions[address] = self.random_node_position()

    def should_show_connection(self, traffic_data, router_time):
        return (
            router_time <
            traffic_data['time'] +
            + BASE_CONNECTION_DISPLAY_DURATION +
            self.get_additional_connection_duration(traffic_data['length'])
        )

    def get_additional_connection_duration(self, length):
        """Gets duration based on number of chars
        using a curve that asymptotes.
        """
        return (
            (
                (
                    length /
                    (length ** 2)
                )
                * CONNECTION_DISPLAY_CURVE_SCALE
            ) + CONNECTION_DISPLAY_CURVE_ADDEND
        ) * CONNECTION_DISPLAY_DURATION_SCALE


    @staticmethod
    def random_node_position():
        return np.array(
            [
                random.randint(
                    NODES_MARGIN,
                    DISPLAY_WIDTH - NODES_MARGIN
                ),
                random.randint(
                    NODES_MARGIN,
                    DISPLAY_HEIGHT - NODES_MARGIN
                ),
                0
            ],
            dtype=np.float32
        )


class NetworkVisualisation:
    """Encapsulates the Glumpy objects and display logic."""

    node_position_manger = None
    router_clock = None

    node_count = 0

    window = None
    master_joins = None

    markers = None
    segments = None

    node_positions = None

    connections = None
    connection_sources = None
    connection_destinations = None

    joins = None
    join_sources = None
    join_destinations = None

    def __init__(self):
        self.node_position_manger = NodePositionManager()
        self.router_clock = RouterClock()

    def run(self):
        app.use(OPEN_GL_BACKEND)

        self.window = app.Window(
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            color=(0, 0, 0, 1),
            fullscreen=FULL_SCREEN,
        )
        self.master_joins = self.get_master_joins(MAXIMUM_NODES)
        self.transform = OrthographicProjection(Position(), aspect=None)
        self.viewport = Viewport()

        @self.window.event
        def on_draw(dt):
            self.update()

        self.initialise()

    def stop(self):
        self.window.close()

    def initialise(self):
        traffic_data = self.get_node_traffic_data()
        self.initialise_graph_state(traffic_data)
        self.initialise_markers()
        self.initialise_segments()
        app.run()

    def initialise_graph_state(self, traffic_data):
        self.node_count = len(traffic_data)
        self.node_positions = self.node_position_manger.get_positions(
            traffic_data
        )
        self.connections = self.node_position_manger.get_connections(
            traffic_data,
            self.router_clock.time
        )
        self.connection_sources, self.connection_destinations = np.nonzero(
            self.connections
        )
        self.joins = self.master_joins[:self.node_count, :self.node_count]
        self.join_sources, self.join_destinations = np.nonzero(self.joins)

    def update(self):
        traffic_data = self.get_node_traffic_data()
        if traffic_data:
            connections_changed, node_count_changed = self.update_graph_state(
                traffic_data
            )

            if node_count_changed:
                self.initialise_markers()

            if connections_changed:
                self.initialise_segments()

            self.stabilise_graph()
            self.window.clear()
            self.segments.draw()
            self.markers.draw()

            if connections_changed or node_count_changed:
                app.run()

    def update_graph_state(self, traffic_data):
        latest_node_count = len(traffic_data)

        self.node_positions = self.node_position_manger.get_positions(
            traffic_data
        )

        connections_changed = False
        new_connections = self.node_position_manger.get_connections(
            traffic_data,
            self.router_clock.time
        )
        if (
                self.connections is not None
                and not np.array_equal(new_connections, self.connections)
        ):
            self.connections = new_connections
            self.connection_sources, self.connection_destinations = np.nonzero(
                self.connections
            )
            connections_changed = True

        self.joins = self.master_joins[:latest_node_count, :latest_node_count]
        self.join_sources, self.join_destinations = np.nonzero(self.joins)

        node_count_changed = False
        if self.node_count != latest_node_count:
            self.node_count = latest_node_count
            node_count_changed = True

        return connections_changed, node_count_changed

    def initialise_markers(self):

        self.markers = collections.MarkerCollection(
            marker='disc',
            transform=self.transform,
            viewport=self.viewport
        )
        self.markers.append(
            self.node_positions,
            size=NODE_SIZE,
            linewidth=0,
            itemsize=1,
            fg_color=(1, 1, 1, 1),
            bg_color=(1, 1, 1, 1)
        )

        self.window.attach(self.transform)
        self.window.attach(self.viewport)

    def initialise_segments(self):
        self.segments = collections.SegmentCollection(
            'agg',
            transform=self.transform,
            viewport=self.viewport
        )
        if np.count_nonzero(self.connections):
            self.segments.append(
                self.node_positions[self.connection_sources],
                self.node_positions[self.connection_destinations],
                linewidth=EDGE_WIDTH,
                itemsize=1,
                color=(1, 1, 1, 1)
            )
        else:
            # Add dummy segments
            self.segments.append(
                np.array([np.array([0,0,0], dtype=np.float32)]),
                np.array([np.array([0,0,0], dtype=np.float32)]),
                linewidth=EDGE_WIDTH,
                itemsize=1,
                color=(0.0, 0.0, 0.0, 1)
            )

        self.window.attach(self.transform)
        self.window.attach(self.viewport)

    def stabilise_graph(self):
        """Ensures stable relative positioning of nodes."""
        positions_x = self.node_positions[:, 0]
        positions_y = self.node_positions[:, 1]
        positions = self.node_positions[:, :2]

        # Global nodes centering
        center_x, center_y = self.window.width/2, self.window.height/2
        positions += 0.01 * ([center_x, center_y] - positions)

        nodes_count = len(self.node_positions)

        if nodes_count > 1:
            # Linked nodes attraction
            distances = (
                    self.node_positions[self.join_sources] -
                    self.node_positions[self.join_destinations]
            )
            L = np.maximum(np.sqrt((distances*distances).sum(axis=1)),1)
            L = (L - NODE_DISTANCE)/L
            distances *= ATTRACTION * L[:,np.newaxis]

            positions_x -= 0.5 * np.bincount(
                self.join_sources,
                distances[:, 0],
                minlength=nodes_count
            )
            positions_y -= .5 * np.bincount(
                self.join_sources,
                distances[:, 1],
                minlength=nodes_count
            )
            positions_x += 0.5 * np.bincount(
                self.join_destinations,
                distances[:, 0],
                minlength=nodes_count
            )
            positions_y += 0.5 * np.bincount(
                self.join_destinations,
                distances[:, 1],
                minlength=nodes_count
            )

            # Global nodes repulsion
            dist = np.maximum(cdist(positions, positions, 'sqeuclidean'), 1)
            distances = np.empty((nodes_count, nodes_count, 2))
            distances[..., 0] = np.subtract.outer(positions_x,positions_x) / dist
            distances[..., 1] = np.subtract.outer(positions_y,positions_y) / dist
            distance_sums = distances.sum(axis=1)
            positions += (
                    REPULSION * distance_sums
                    / np.sqrt(((distance_sums * distance_sums).sum(axis=0)))
            )

        # Update self.markers and self.segments
        self.markers["position"] = self.node_positions

        if np.count_nonzero(self.connections):
            # Update segment positions if any connections active
            self.segments["P0"] = np.repeat(
                self.node_positions[self.connection_sources], 4, axis=0
            )
            self.segments["P1"] = np.repeat(
                self.node_positions[self.connection_destinations], 4, axis=0
            )

        # Update node position manager
        self.node_position_manger.set_positions(self.node_positions)

    @staticmethod
    def get_node_traffic_data():
        return json.loads(redis_client.get('traffic').decode())

    @staticmethod
    def get_master_joins(max_nodes):
        return np.array(
            [
                [0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
                [1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            ]
        )



