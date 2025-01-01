#include <OneWire.h>
#include <DallasTemperature.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

// Data wire is conntec to the Arduino digital pin 4
#define ONE_WIRE_BUS 4

// Setup a oneWire instance to communicate with any OneWire devices
OneWire oneWire(ONE_WIRE_BUS);

// Pass our oneWire reference to Dallas Temperature sensor
DallasTemperature sensors(&oneWire);

#define VIB_1 3
#define VIB_2 2

#define BUZZ 7

float tempVal;
long lastUpdate;

byte degree[] = {
  B11100,
  B10100,
  B11100,
  B00000,
  B00000,
  B00000,
  B00000,
  B00000
};

char BTData = '0';

void setup(void)
{
  // Start serial communication for debugging purposes
  Serial.begin(9600);

  lcd.init();
  lcd.createChar(0, degree);
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("   Drawsiness   ");
  lcd.setCursor(0, 1);
  lcd.print("   Detection   ");

  pinMode(VIB_1, OUTPUT);
  pinMode(VIB_2, OUTPUT);
  pinMode(BUZZ, OUTPUT);

  digitalWrite(VIB_1, HIGH);
  digitalWrite(VIB_2, HIGH);

  digitalWrite(BUZZ, LOW);

  // Start up the library
  sensors.begin();

  delay(3000);
}

void loop(void) {
  tempVal = getTemperature();
  readSerialMonitor();

  if (tempVal > 100) {
    alertTemp();
  }

  if (millis() - lastUpdate > 1000) {
    updateLcd();
  }

  delay(200);
}

float getTemperature() {
  // Call sensors.requestTemperatures() to issue a global temperature and Requests to all devices on the bus
  sensors.requestTemperatures();
  //  Serial.print(" - Fahrenheit temperature: ");
  //  Serial.println(sensors.getTempFByIndex(0));
  return sensors.getTempFByIndex(0);
}

void startVibrate() {
  digitalWrite(VIB_1, HIGH);
  digitalWrite(VIB_2, LOW);
  //  Serial.println("Vibrating");
  //  for (int i = 0; i < times; i++) {
  //    digitalWrite(VIB_1, HIGH);
  //    digitalWrite(VIB_2, LOW);
  //    delay(delayTime);
  //    digitalWrite(VIB_1, LOW);
  //    digitalWrite(VIB_2, LOW);
  //  }
}

void stopVibrate() {
  digitalWrite(VIB_1, LOW);
  digitalWrite(VIB_2, LOW);
}

void readSerialMonitor() {
  if (Serial.available() > 0) {
    BTData = Serial.read();
    
  }
  if (BTData == '1') {
    beep(5, 200);
    alertMessage();
    startVibrate();
  }else if (BTData == '2') {
    beep(5, 200);
    alertDraw();
    startVibrate();
  } else if (BTData == '0') {
    stopVibrate();
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("MONITORING......");
  }
}

void alertMessage() {
  lcd.setCursor(0, 0);
  lcd.print("ALERT  PAY  ATTN");
}

void alertDraw(){
  lcd.setCursor(0, 0);
  lcd.print("DRAWSINESS DTECD");
}

void alertTemp() {
  lcd.setCursor(0, 0);
  lcd.print("BODY TEMP INCRSD");
  lcd.setCursor(0, 1);
  lcd.print(" DO  NOT  DRIVE ");
  beep(5, 200);
}

void updateLcd() {
  lcd.setCursor(0, 1);
  lcd.print("Bdy Tp: ");
  lcd.print(tempVal, 2);
  lcd.write(byte(0));
  lcd.print("F");
}

void beep(int times, int delayTime) {
  for (int i = 0; i < times; i++) {
    digitalWrite(BUZZ, HIGH);
    delay(delayTime);
    digitalWrite(BUZZ, LOW);
    delay(delayTime);
  }
}
