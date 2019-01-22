import time
import pymongo
from flask import Flask
import museum_v2


app = Flask(__name__)
# cache = redis.Redis(host='redis', port=6379)


# def get_hit_count():
#     retries = 5
#     while True:
#         try:
#             return cache.incr('hits')
#         except redis.exceptions.ConnectionError as exc:
#             if retries == 0:
#                 raise exc
#             retries -= 1
#             time.sleep(0.5)


@app.route('/')
def hello():
    museum_v2.main()
    return 'Hello World! I have been seen 1 times.\n'

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)