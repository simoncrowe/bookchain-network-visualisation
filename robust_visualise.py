import configparser
import os
import redis
import requests
import threading

from visualise import NetworkVisualisation

redis_client = redis.Redis()


MODULE_PARENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

config = configparser.ConfigParser()
config.read(os.path.join(MODULE_PARENT_DIRECTORY, 'config.ini'))


QUEUE_ROUTER_HOST = config['QUEUE_ROUTER']['host']
QUEUE_ROUTER_TOKEN = config['QUEUE_ROUTER']['token']
TRAFFIC_API_CALL_INTERVAL = float(
    config['QUEUE_ROUTER']['traffic_api_call_interval']
)


def update_node_traffic_cache():
    redis_client.set(
        'traffic',
        requests.get(
            'https://{host}/traffic?token={token}'.format(
                host=QUEUE_ROUTER_HOST,
                token=QUEUE_ROUTER_TOKEN,
            )
        ).content
    )
    threading.Timer(
        TRAFFIC_API_CALL_INTERVAL,
        update_node_traffic_cache
    ).start()


update_node_traffic_cache()


while True:
    visualisation = NetworkVisualisation()
    try:
        visualisation.run()
    except:
        visualisation.stop()
        del visualisation
    else:
        break