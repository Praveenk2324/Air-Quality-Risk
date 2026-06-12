import json
import random
import time

from kafka import KafkaProducer

TOPIC = 'air-quality-readings'
BOOTSTRAP_SERVERS = ['localhost:9092']

POLLUTANT_RANGES = {
    'pm25': (2.0, 200.0),
    'no2': (5.0, 120.0),
    'o3': (10.0, 180.0)
}

CITIES = ['Bengaluru', 'Delhi', 'Mumbai', 'Chennai', 'Hyderabad']

def make_readings() -> dict:
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        'city': random.choice(CITIES),
        'pm25': round(random.uniform(*POLLUTANT_RANGES['pm25']), 2),
        'no2': round(random.uniform(*POLLUTANT_RANGES['no2']), 2),
        'o3': round(random.uniform(*POLLUTANT_RANGES['o3']), 2),
        'hour_of_day': now.hour,
        'month': now.month,
        'timestamp': now.isoformat(),
    }

def main(interval_seconds: float = 1):
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    )
    print(f"Streaming readings to topic '{TOPIC}' every {interval_seconds}s ...")
    try:
        while True:
            reading = make_readings()
            producer.send(TOPIC, value=reading)
            print(f" Sent: {reading}")
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print('Stopped.')
    finally:
        producer.close()

if __name__=='__main__':
    main()