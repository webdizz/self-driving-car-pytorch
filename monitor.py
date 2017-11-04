from influxdb import InfluxDBClient
import datetime


def track(dbclient, topic, value):
    # Use utc as timestamp
    receiveTime = datetime.datetime.utcnow()
    base_topic = 'racer'
    metric = [
        {
            "measurement": "{}.{}".format(base_topic, topic),
            "time": receiveTime,
            "fields": {
                "value": float(value)
            }
        }
    ]
    dbclient.write_points(metric)


# Set up a client for InfluxDB
dbclient = InfluxDBClient(
    'localhost', 8086, 'influxdb', 'influxdb', 'influxdb')
