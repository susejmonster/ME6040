class Flock {
  ArrayList<Module> mods; // An ArrayList for all the boids

  Flock() {
    mods = new ArrayList<Module>(); // Initialize the ArrayList
  }

  void run() {
    for (Module b : mods) {
      b.run(mods);  // Passing the entire list of boids to each boid individually
    }
  }

  void addBoid(Module b) {
    mods.add(b);
  }

}

class Module {
  int xOffset;
  int yOffset;
  float x, y;
  int unit;
  int xtheta = 1;
  int ytheta = 1;
  float speed; 
  PVector position;
  PVector velocity;
  PVector acceleration;
  float r;
  float maxforce;    // Maximum steering force
  float maxspeed;    // Maximum speed
  
  // Contructor
  Module(int xOffsetTemp, int yOffsetTemp, int xTemp, int yTemp, float speedTemp, int tempUnit) {
    xOffset = xOffsetTemp;
    yOffset = yOffsetTemp;
    x = xTemp;
    y = yTemp;
    speed = speedTemp;
    unit = tempUnit;
    float angle = random(TWO_PI);
    velocity = new PVector(cos(angle), sin(angle));
    position = new PVector(x, y);
    r = 2.0;
    maxspeed = 2;
    maxforce = 0.03;
    acceleration = new PVector(0, 0);
  }
  
  //!!!update velocities and BCS!!!//
  void update() {
     PVector steering = align(mods);
  
  // 2. Add noise by rotating the vector slightly
  
    steering.rotate(random(-0.1, 0.1)); 
  
  // 3. Apply the steering to your acceleration
    acceleration.add(steering);
  
  // 4. Standard movement math
    velocity.add(acceleration);
    velocity.limit(maxspeed);
    position.add(velocity);
  
  // 5. Update your x/y variables for the display() function
    x = position.x;
    y = position.y;
  
  // 6. Reset acceleration for the next frame
  acceleration.mult(0);
    
    if (position.x < -r) position.x = width+r;
    if (position.y < -r) position.y = height+r;
    if (position.x > width+r) position.x = -r;
    if (position.y > height+r) position.y = -r;  
  }
   
   
  //!!!average direction of velocity calculator!!!//
  PVector align (ArrayList<Module> mods) {
    float neighbordist = 50; // The radius to look for neighbors
    PVector sum = new PVector(0, 0);
    int count = 0;

    for (Module other : mods) {
      float d = PVector.dist(position, other.position);
      if ((d > 0) && (d < neighbordist)) {
        sum.add(other.velocity); // Add the direction/speed of the neighbor
        count++;
      }
    }

    if (count > 0) {
      sum.div((float)count); // The average velocity vector
      return sum;
    } else {
    // If alone, the "average direction" is just where I'm already going
      return velocity.copy(); 
    }
  }
  
  //apply alignment to all//
  void flock(ArrayList<Module> mods) {
     align(mods);      // Alignment  
  }
  //apply alignment to all//
  void run(ArrayList<Module> mods) {
    flock(mods);
  }

   
   
  // Custom method for drawing the object
  void display() {
    fill(255);
    ellipse(x+xOffset, y+yOffset, 6, 6);
  }
}

/**
 * Array Objects. 
 * 
 * Demonstrates the syntax for creating an array of custom objects. 
 */

int unit = 10;
int count;
ArrayList<Module> mods;
int mouse_xrad = 50;
int mouse_yrad = 50;
void setup() {
  size(640, 360);
  mods = new ArrayList<Module>(); // Initialize as ArrayList

  int unit = 10;
  int wideCount = width / unit;
  int highCount = height / unit;

  for (int y = 0; y < highCount; y++) {
    for (int x = 0; x < wideCount; x++) {
      // Use .add() instead of [index++]
      mods.add(new Module(x*unit, y*unit, unit/2, unit/2, random(0.05, 0.8), unit));
    }
  }
}

void draw() {
  background(0);
  for (Module mod : mods) {
    mod.update();
    mod.display();
  }
  
  
  pushStyle(); 
    stroke(#FF0000);       // Changed to 255 (white) so it's visible on black
    strokeWeight(2);  // Thick outline
    noFill();          // Transparent center
    ellipse(mouseX, mouseY, mouse_xrad, mouse_yrad);
  popStyle();
}

void mouseWheel(MouseEvent event) {
  float e = event.getCount();
  println(e);
  if (e>0) {
      mouse_xrad=mouse_xrad+1;
      mouse_yrad=mouse_yrad+1;
    }
   if (e<0) {
      mouse_xrad=mouse_xrad-1;
      mouse_yrad=mouse_yrad-1;
    }
  
}
