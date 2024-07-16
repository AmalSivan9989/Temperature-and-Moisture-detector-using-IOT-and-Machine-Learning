#include <DHT.h>

#define DHTPIN 3     // Pin where the DHT11 is connected
#define DHTTYPE DHT11   // DHT 11
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  dht.begin();
}

void loop() {
  delay(2000); // Wait a few seconds between measurements

  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }

  // Send temperature and humidity in the format "temp,humidity"
  Serial.print(temperature);
  Serial.print(",");
  Serial.println(humidity);
}
