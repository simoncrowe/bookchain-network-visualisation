"""Consumes queue-router's /traffic API and outputs an OpenGL visualisation
Adapted from: https://github.com/glumpy/glumpy/blob/master/examples/graph.py
"""

from collections import OrderedDict
import configparser
from datetime import time
import os
import random

from glumpy import app, collections
from glumpy.transforms import Position, OrthographicProjection, Viewport
import numpy as np
from scipy.spatial.distance import cdist

import requests


MODULE_PARENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SHADERS_DIRECTORY = os.path.join(MODULE_PARENT_DIRECTORY, 'shaders')

config = configparser.ConfigParser()
config.read(os.path.join(MODULE_PARENT_DIRECTORY, 'config.ini'))

QUEUE_ROUTER_HOST = config['QUEUE_ROUTER']['host']
QUEUE_ROUTER_PORT = config['QUEUE_ROUTER']['port']
QUEUE_ROUTER_TOKEN = config['QUEUE_ROUTER']['token']

FULL_SCREEN = config['DISPLAY']['full_screen'] == 'True'
DISPLAY_WIDTH = int(config['DISPLAY']['width'])
DISPLAY_HEIGHT = int(config['DISPLAY']['height'])
NODE_SPACING = float(config['DISPLAY']['node_spacing'])
NODES_MARGIN = float(config['DISPLAY']['nodes_margin'])

MAXIMUM_NODES = int(config['NETWORK']['maximum_nodes'])
NODE_SIZE = int(config['NETWORK']['node_size'])
EDGE_WIDTH = float(config['NETWORK']['edge_width'])
ATTRACTION = float(config['NETWORK']['attraction'])
REPULSION = float(config['NETWORK']['repulsion'])
NODE_DISTANCE = float(config['NETWORK']['node_distance'])


class NodePositionManager:
    """Manages addressed node positions based on router traffic data."""

    positions = OrderedDict()

    def get_positions(self, traffic_data):
        self.prune_inactive(traffic_data)
        self.create_any_new(traffic_data)
        return np.array(list(self.positions.values()))

    def set_positions(self, position_array):
        for index, address in enumerate(self.positions.keys()):
            self.positions[address] = position_array[index]

    def prune_inactive(self, traffic_data):
        for address in self.positions:
            if address not in traffic_data:
                del self.positions[address]

    def create_any_new(self, traffic_data):
        for address in traffic_data:
            if address not in self.positions:
                self.positions[address] = next(self.valid_node_position())

    def valid_node_position(self):
        for _ in range(1024):
            _position = np.array(
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
            correctly_spaced = True

            for position in self.positions.values():
                if self.vector_distance(position, _position) < NODE_SPACING:
                    correctly_spaced = False
                    break

            if correctly_spaced:
                yield _position

        raise RuntimeError(
            'Unable to find space for new node. '
            'Consider setting node_spacing lower.'
        )

    @staticmethod
    def vector_distance(vector_a, vector_b):
        return np.linalg.norm(vector_a - vector_b)


class NetworkVisualisation:
    """Encapsulates the Glumpy objects and display logic."""

    node_position_manger = NodePositionManager()
    node_count = 0

    window = None
    master_connections = None

    markers = None
    segments = None
    node_positions = None
    connections = None
    sources = None
    destinations = None

    def run(self):
        self.window = app.Window(
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            color=(0, 0, 0, 1),
            fullscreen=FULL_SCREEN,
        )
        self.master_connections = self.get_master_connections(MAXIMUM_NODES)

        @self.window.event
        def on_draw(dt):
            self.update()

        app.run()

    def update(self):
        print('Calling update at ' + str(time()))
        traffic_data = self.get_node_traffic_data()
        if traffic_data:
            self.set_graph_state(traffic_data)
            self.stabilise_graph()
            self.window.clear()
            self.segments.draw()
            self.markers.draw()

    def initialise(self, traffic_data):

        transform = OrthographicProjection(Position(), aspect=None)
        viewport = Viewport()

        self.markers = collections.MarkerCollection(
            marker='disc',
            transform=transform,
            viewport=viewport
        )
        self.markers.append(
            self.node_positions,
            size=NODE_SIZE,
            linewidth=0,
            itemsize=1,
            fg_color=(1, 1, 1, 1),
            bg_color=(1, 1, 1, 1)
        )
        self.segments = collections.SegmentCollection(
            'agg',
            transform=transform,
            viewport=viewport
        )
        self.segments.append(
            self.node_positions[self.sources],
            self.node_positions[self.destinations],
            linewidth=EDGE_WIDTH,
            itemsize=1,
            color=(0.5, 0.5, 0.5, 1)
        )

        self.window.attach(transform)
        self.window.attach(viewport)

    def set_graph_state(self, traffic_data):
        latest_node_count = len(traffic_data)

        self.node_positions = self.node_position_manger.get_positions(
            traffic_data
        )
        self.connections = self.master_connections[
           :latest_node_count, :latest_node_count
        ]

        self.sources, self.destinations = np.nonzero(self.connections)

        if self.node_count != latest_node_count:
            self.initialise(traffic_data)
            self.node_count = latest_node_count

    def stabilise_graph(self):
        """Ensures stabled relative positioning of nodes.
        """
        positions_x = self.node_positions[:, 0]
        positions_y = self.node_positions[:, 1]
        positions = self.node_positions[:, :2]

        # Global nodes centering
        center_x, center_y = self.window.width/2, self.window.height/2
        positions += 0.01 * ([center_x, center_y] - positions)

        # Linked nodes attraction
        distances = (
            self.node_positions[self.sources] - self.node_positions[self.destinations]
        )
        L = np.maximum(np.sqrt((distances*distances).sum(axis=1)),1)
        L = (L - NODE_DISTANCE)/L
        distances *= ATTRACTION * L[:,np.newaxis]
        nodes_count = len(self.node_positions)

        positions_x -= 0.5 * np.bincount(
            self.sources,
            distances[:, 0],
            minlength=nodes_count
        )
        positions_y -= .5 * np.bincount(
            self.sources,
            distances[:, 1],
            minlength=nodes_count
        )
        positions_x += 0.5 * np.bincount(
            self.destinations,
            distances[:, 0],
            minlength=nodes_count
        )
        positions_y += 0.5 * np.bincount(
            self.destinations,
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
        self.segments["P0"] = np.repeat(
            self.node_positions[self.sources], 4, axis=0
        )
        self.segments["P1"] = np.repeat(
            self.node_positions[self.destinations], 4, axis=0
        )

        # Update node position manager
        self.node_position_manger.set_positions(self.node_positions)

    @staticmethod
    def get_node_traffic_data():
        return requests.get(
            'http://{host}:{port}/traffic?token={token}'.format(
                host=QUEUE_ROUTER_HOST,
                port=QUEUE_ROUTER_PORT,
                token=QUEUE_ROUTER_TOKEN,
            )
        ).json()

    @staticmethod
    def get_router_time():
        return requests.get(
            'http://{host}:{port}/time?token={token}'.format(
                host=QUEUE_ROUTER_HOST,
                port=QUEUE_ROUTER_PORT,
                token=QUEUE_ROUTER_TOKEN,
            )
        ).json()

    @staticmethod
    def get_master_connections(max_nodes):
        return np.array(
            [
                [random.randrange(2) for _ in range(max_nodes)]
                for i in range(max_nodes)
            ]
        )


visualisation = NetworkVisualisation()
visualisation.run()
