#include <QTRSensors.h>
// 1,2,3 sağda 4,5,6 sol A SOL
#define BASE 125
#define PWMB 9
#define PWMA 11
#define AIN1 6
#define AIN2 7
#define BIN1 4
#define BIN2 5
#define kp 0.025f//0.025f
#define kd 0.001f//0.001f
#define k 0.7f//exponential pid constant values (0,1]  1 means regular pid
#define emitter_pin 8
#define encoderA 2
#define encoderB 3
#define turn_speed 80
#define thresh 200


#define ARDUINO_OK 10
#define RASPBERRY_OK 11

#define READ_SIGN 19
 
//line_type= b: black line w: white line
float motorSpeedA,motorSpeedB;
uint16_t position;
double last_pid=0,pid,exp_pid,lasterror=0,error;
char line_type='b',image='r';//image= r: right l: left s: straight A1: a A2: b A3: c B1: d B2: e B3: f
QTRSensors qtr;
bool change=0;//if line changed return 1
const uint8_t SensorCount = 6;
uint16_t sensorValues[SensorCount];
char park_side;//A is left B is right
int intended_slot;//i.e. 1,2,3
int current_slot=1;//şuandaki dönüş


void setup()
{
  Serial.begin(9600);
  // configure the sensors
  qtr.setTypeAnalog();
  qtr.setSensorPins((const uint8_t[]){A0, A1, A2, A3, A4, A5}, SensorCount);
  qtr.setEmitterPin(emitter_pin);

  pinMode(PWMA,OUTPUT);
  pinMode(PWMB ,OUTPUT);
  pinMode(AIN1,OUTPUT);
  pinMode(AIN2,OUTPUT);
  pinMode(BIN1,OUTPUT);
  pinMode(BIN2 ,OUTPUT);
 
  
  delay(500);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH); // turn on Arduino's LED to indicate we are in calibration mode


  for (uint16_t i = 0; i < 400; i++)
  {
    qtr.calibrate();
  }
  digitalWrite(LED_BUILTIN, LOW); // turn off Arduino's LED to indicate we are through with calibration

  // print the calibration minimum values measured when emitters were on
  
  //for (uint8_t i = 0; i < SensorCount; i++)
 // {
    //Serial.print(qtr.calibrationOn.minimum[i]);
    //Serial.print(' ');
  //}
  //Serial.println();

  // print the calibration maximum values measured when emitters were on
  /*for (uint8_t i = 0; i < SensorCount; i++)
  {
    Serial.print(qtr.calibrationOn.maximum[i]);
    Serial.print(' ');
  }
  Serial.println();
  Serial.println();*/
  delay(1000);
  initialize();
  
}

void loop()
{
  read_sens();
  motor_drive(); 
}
/*void printt()
{
for (uint8_t i = 0; i < SensorCount; i++)
  {
    Serial.print(sensorValues[i]);
    Serial.print('\t');
  }
  Serial.print(position);
  Serial.print('\t');
    Serial.print(motorSpeedA);
  Serial.print('\t');
    Serial.print(motorSpeedB);
  Serial.print('\t');
    Serial.println(pid);
    delay(100);
} for debugging purposes*/

void initialize(){

  delay(1000);
  Serial.write(ARDUINO_OK);

  while(Serial.read() != 'r');

  //Serial.println("read!");

  flushIncoming();
  Serial.flush();
}

void flushIncoming(){
  while(Serial.available())
  Serial.read();
}

char readSign(){ 
    //send read signal , then halt

    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, LOW);
    analogWrite(PWMA, 0);
    digitalWrite(BIN1, LOW);
    digitalWrite(BIN2, LOW);
    analogWrite(PWMB, 0);
    
    Serial.write(READ_SIGN);

    while(!Serial.available());
    image = Serial.read();
   
 
}

void read_sens()
{
  change=0;
  if(line_type=='b')
   {
       position = qtr.readLineWhite(sensorValues);
   }
   else
   {
       position = qtr.readLineBlack(sensorValues);
   }
   bool right_path=check_path_right();
   bool left_path=check_path_left();
   if(left_path&&right_path)
   {

      if(sensorValues[2]<200||sensorValues[3]<200)
      {
          change=1;
          line_type='w';
          readSign();
          if(image=='a'||image=='b'||image=='c'||image=='d'||image=='e'||image=='f')
           park(image);
      }
      else
      {
        
        // Şuraya park mark eklencek
        switch(image)
        {
          case 'l':
          turn_left();
          image='s';
          break;
          case 'r':
          turn_right();
          image='s';
          break;
          default:;
        }
 
      }
   }
   else if(left_path||right_path)
   {
    
    if(left_path)
      {
          
          if(image=='l')
          {
           turn_left();
           image='s';
          }
      }
      else 
      {
         
         if(image=='r')
         {
          turn_right();
          image='s';
         }
      }
      
   }
   else
   {
      if(line_type=='w')
        change=1;
      line_type='b';
   }


}
bool check_path_right()//right 1 else 0
{
if((sensorValues[0]>=200&&sensorValues[1]>=200&&sensorValues[2]>=200)||(sensorValues[0]>=200&&sensorValues[1]))
return 1;
else
return 0;
}
bool check_path_left()//left 1 else 0
{
if((sensorValues[5]>=200&&sensorValues[4]>=200&&sensorValues[3]>=200)||(sensorValues[5]>=200&&sensorValues[4]))
return 1;
else
return 0;
}
void motor_drive()
{
if(change==1)
{
  motorSpeedA=BASE;
  motorSpeedB=BASE;
  change=0;
}
  
else if (position>=2500)
{
    error=position-2500; 
    pid = kp * error + kd *(error-lasterror); 
    exp_pid=k*pid+(1-k)*last_pid;
    motorSpeedA = BASE + exp_pid;
    motorSpeedB = BASE - exp_pid;
}
else
{
    error=2500-position;
    pid = kp * error + kd *(error-lasterror);
    exp_pid=k*pid+(1-k)*last_pid;
    motorSpeedA = BASE - exp_pid;
    motorSpeedB = BASE + exp_pid;

}

if(motorSpeedA<0)
motorSpeedA=0;
if(motorSpeedB<0)
motorSpeedB=0;

digitalWrite(AIN1, HIGH);
digitalWrite(AIN2, LOW);
analogWrite(PWMA, motorSpeedA);
digitalWrite(BIN1, HIGH);
digitalWrite(BIN2, LOW);
analogWrite(PWMB, motorSpeedB);

lasterror=error;
last_pid=pid;
}
void turn_right()
{
  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, LOW);
  analogWrite(PWMA, 0);
  digitalWrite(BIN1, LOW);
  digitalWrite(BIN2, LOW);
  analogWrite(PWMB, 0);
  delay(40);
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);
  analogWrite(PWMA, 80);
  digitalWrite(BIN1, HIGH);
  digitalWrite(BIN2, LOW);
  analogWrite(PWMB, 80);
  delay(200);
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);
  analogWrite(PWMA, turn_speed);
  digitalWrite(BIN1, LOW);
  digitalWrite(BIN2, HIGH);
  analogWrite(PWMB, turn_speed);
  delay(200);
  while(analogRead(A2)<thresh);
}
void turn_left()
{
  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, LOW);
  analogWrite(PWMA, 0);
  digitalWrite(BIN1, LOW);
  digitalWrite(BIN2, LOW);
  analogWrite(PWMB, 0);
  delay(40);
  digitalWrite(AIN1, HIGH);
  digitalWrite(AIN2, LOW);
  analogWrite(PWMA, 80);
  digitalWrite(BIN1, HIGH);
  digitalWrite(BIN2, LOW);
  analogWrite(PWMB, 80);
  delay(200);
  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, HIGH);
  analogWrite(PWMA, turn_speed);
  digitalWrite(BIN1, HIGH);
  digitalWrite(BIN2, LOW);
  analogWrite(PWMB, turn_speed);
  delay(200);
  while(analogRead(A3)<thresh);
}
void park(char parking_index)//parking_index = A1: a A2: b A3: c B1: d B2: e B3: f
{
  transform(parking_index);
  while(1)
  {
    read_sens_park();
    motor_drive();
  }
}
void transform(char parking_index)//fill in parking_side and intended_slot usage at line 28,29
{
switch(parking_index)
{
  case 'a':
  park_side='A';
  intended_slot=1;
  break;
  case 'b':
  park_side='A';
  intended_slot=2;
  break;
  case 'c':
  park_side='A';
  intended_slot=3;
  break;
  case 'd':
  park_side='B';
  intended_slot=1;
  break;
  case 'e':
  park_side='B';
  intended_slot=2;
  break;
  case 'f':
  park_side='B';
  intended_slot=3;
  break;
  default:;
}
}
void read_sens_park()
{
  change=0;
  if(line_type=='b')
   {
       position = qtr.readLineWhite(sensorValues);
   }
   else
   {
       position = qtr.readLineBlack(sensorValues);
   }
   bool right_path=check_path_right();
   bool left_path=check_path_left();
   if(left_path&&right_path)
   {

      if(sensorValues[2]<200||sensorValues[3]<200)
      {
          if(line_type=='b')
          change=1;
          line_type='w';
          
          //iletişim kur imagea data ver kullanım -> line 20
      }
      else
      {
        if(current_slot<intended_slot)
        {
            current_slot++;
            delay(100);
        }
        else if(current_slot==intended_slot)
        {
            current_slot++;
            if(park_side=='A')
            {
              turn_left();
            }
            else
            {
              turn_right();
            }
        }
        else 
        {
          Stop(20000);
        }
      }
   }
   
   else
   {
      if(line_type=='w')
        change=1;
      line_type='b';
   }


}
void Stop(int a)
{
  digitalWrite(AIN1, LOW);
  digitalWrite(AIN2, LOW);
  analogWrite(PWMA, 0);
  digitalWrite(BIN1, LOW);
  digitalWrite(BIN2, LOW);
  analogWrite(PWMB, 0);
  delay(a);
}
