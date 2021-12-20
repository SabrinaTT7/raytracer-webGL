function RayTracer(canvasID) {
  /**
   * Initializes an instance of the RayTracer class.
   * You may also wish to set up your camera here.
   * (feel free to modify input parameters to set up camera).
   *
   * @param canvasID (string) - id of the canvas in the DOM where we want to render our image
   */
  // setup the canvas
  this.canvas = document.getElementById(canvasID);


  // setup the background style: current options are 'daylight' or 'white'
  this.sky = 'daylight';

  this.aliasing = false;
  //initialize to debugging mode
  this.index = 1;

  // initialize the objects and lights
  this.objects = new Array();
  this.lights  = new Array();

  //set up the camera
  //hardcoded for now
    let cam_params = {
      'eye': vec3.fromValues(0, 0, 10),
      'center': vec3.fromValues(0, 0, 0),
      'up': vec3.fromValues(0, 1, 0),
      'fov': Math.PI/6,
      'aspect': this.canvas.width/this.canvas.height //change this to calculate based on canvas
    }
  let cam = new Camera(cam_params);
  cam.setUp();
  this.camera = cam;
}

RayTracer.prototype.draw = function() {
  /**
   * Renders the scene to the canvas.
   * Loops through all pixels and computes a pixel sample at each pixel midpoint.
   * Pixel color should then be computed and assigned to the image.
  **/
  // get the canvas and the image data we will write to
  let context = this.canvas.getContext('2d');
  let image = context.createImageData(this.canvas.width,this.canvas.height);


  //setup the image stuff for the background image
  let background_image = document.getElementById('background');
  let img_canvas = document.createElement('canvas');
  let cntx = img_canvas.getContext('2d');
  img_canvas.width =  background_image.width;
  img_canvas.height = background_image.height;
  cntx.drawImage(background_image, 0, 0 );
  let data = cntx.getImageData(0, 0, background_image.width, background_image.height);
  data = data.data;

  this.data = data;
  this.background_image = background_image;

  // numbers of pixels in x- and y- directions
  const nx = image.width;
  const ny = image.height;

  //save these for use in background function
  this.nx = nx;
  this.ny = ny;

  //loop through the objects to initialize the triangles
  for (let o = 0; o < this.objects.length; o++){
    //console.log(this.objects);
    let obj = this.objects[o];

    if (obj.type === 'mesh'){

      let new_objs = obj.setUp();
      this.objects.pop(o);
      for (let i = 0; i < new_objs.length; i ++){
        this.objects.push(new_objs[i]);
      }

    }
   // console.log(this.objects[o]);
  }
  //console.log(this.objects)
   //hardcoded for now
  const tmin = .001;
  const tmax = 100000000000000000000; //ie20 ish
  // loop through the canvas pixels //j < ny, i<  nx
  for (let j = 0; j < ny; j+=this.index) {
    for (let i = 0; i < nx; i+= this.index) {



      // compute pixel color
      let color = vec3.create();
      let color_sum = vec3.fromValues(0,0,0);
      let closest_color = vec3.create();

      //keep track of the closest intersection;
      let range;
      if (this.aliasing ===true){
        //set to number of samples we want
        range = 20;
      }
      else{
        range = 1;
      }


      //for i in range range
      for (let k = 0; k < range; k++){


        let px;
        let py;
        if (k === 0){
          // compute pixel coordinates in [0,1] x [0,1]
          px = (i + 0.5) / nx;      // sample at pixel center
          py = (ny - j - 0.5) / ny; // canvas has y pointing down, but image plane has y going up
        }
        //after that we use math.random
        else{
          let n = Math.random();
          px = (i + n) / nx;      // sample at pixel center
          py = (ny - j - n) / ny;

        }

        let closest = tmax;

        // cast a ray through (px,py) and call some 'color' function
        let ray = this.camera.getRay(px, py);


        //initialize the closest color to be the background color
        vec3.copy(closest_color, this.background(ray));
        //keep track of hits
        let miss = true;
        for (let o = 0; o < this.objects.length; o++){

          let obj = this.objects[o];
          if (obj.type !== 'mesh'){
             //console.log(obj);

            let check = obj.intersect(ray, tmin, tmax);
            if (check !== undefined){
              miss = false;
              //color according to the closest intersection
              if (check.t < closest){
                //call the color function
                new_color = this.color(ray, check, 10);

                vec3.copy(closest_color,  new_color);
                closest = check.t;
              }
            }
          }
        }
        vec3.add(color_sum, color_sum, closest_color);

      }
      //divide the sum by the amount of samples taken
      vec3.scale(color, color_sum, 1/range);

      // set the pixel color into our final image
      this.setPixel( image , i,j , color[0] , color[1] , color[2] );
    }
  }
  context.putImageData(image,0,0);
}
RayTracer.prototype.color = function(ray, hit, depth){

  //if we have reached maximum recursion depth, return background color.
  if (depth === 0){
    return vec3.fromValues(0,0,0);

  }
  let color = vec3.create();

  //save tmin and tmax for convenience
  const tmin = .001;
  const tmax = 100000000000000000000; //ie20 ish

  //calculate La as an average of lights
  let La = vec3.create();
  for (let i = 0; i < this.lights.length; i++){
    vec3.add(La, La, this.lights[i].color);
  }
  vec3.scale(La, La, 1/this.lights.length);
  let ka = vec3.fromValues(.4, .4, .4);
  if (hit.material !== undefined){
    ka = hit.material.ka;
  }
  vec3.multiply(color, La, ka);

  //loop through the lights to determine lighting
  for (let i = 0; i < this.lights.length; i++){
    let blocked = false;
    let dir = vec3.create();
    vec3.sub(dir, this.lights[i].location, hit.intersection);
    let abnormal = vec3.create();
    vec3.copy(abnormal, dir);
    vec3.normalize(dir, dir);
    let shadow_ray = new Ray({'point': hit.intersection, 'direction': dir});
    //save this for backgound image purposes
    shadow_ray.abnormal_direction = abnormal;

    //check if the ray hits any of the objects in the scene
    for (let j = 0; j < this.objects.length; j++){
      if (this.objects[j].type !== 'mesh'){
        let blocking_object = this.objects[j].intersect(shadow_ray, tmin, tmax);
        if (blocking_object !== undefined){
          //if it does, it is in darkness (shadow)
          blocked = true;
        }
      }
    }
    //calculates useing Blinn-Phong
    let shaded_color = hit.material.shade(ray, this.lights[i], hit);

    //if not blocked, add the lighting shading
    if(!blocked){
      vec3.add(color, color, shaded_color);
    }
  }
  //throw scatter rays recursively to calculate reflection and refraction
  let scattered_ray;
  if (hit.material !== undefined){
    scattered_ray = hit.material.scatter(ray, hit);
  }

  if (scattered_ray === undefined){
    return color;
  }
  //calculate new_hit information for this ray
  let color_scattered = vec3.create();
  let closest = tmax;

  let new_hit;
  //loop through all the objects and call the color function recursively
  for (let o = 0; o < this.objects.length; o++){
        let obj = this.objects[o];
        if (obj.type !== "mesh"){
          let new_hit = obj.intersect(scattered_ray, tmin, tmax);
          if (new_hit !== undefined){

            //color according to the closest intersection
            if (new_hit.t < closest){
              //call the color function
              //use new hit information here to replicate what is done in draw
              color_scattered = this.color(scattered_ray, new_hit, depth -1);



              closest = new_hit.t;
            }
          }
        }

      }
    //if we havent hit anything, we have 'hit' the background
    if (closest === tmax){
      color_scattered = this.background(scattered_ray);
    }
  //weight the original colors with the scattered colors according to the strength of the material (default to .7)
  vec3.scale(color_scattered, color_scattered, hit.material.strength );
  vec3.scale(color, color, 1-hit.material.strength);

  vec3.add(color, color, color_scattered);

  return color;

}

RayTracer.prototype.background = function(ray) {
  /**
   * Computes the background color for a ray that goes off into the distance.
   *
   * @param ray - ray with a 'direction' (vec3) and 'point' (vec3)
   * @returns a color as a vec3 with (r,g,b) values within [0,1]
   *
   * Note: this assumes a Ray class that has member variable ray.direction.
   * If you change the name of this member variable, then change the ray.direction[1] accordingly.
  **/
  if (this.sky === 'white') {
    // a white sky
    return vec3.fromValues(1,1,1);
  }
  else if (this.sky === 'daylight') {
    // a light blue sky :)
    let t = 0.5*ray.direction[1] + 0.2; // uses the y-values of ray.direction
    if (ray.direction == undefined) t = 0.2; // remove this if you have a different name for ray.direction
    let color = vec3.create();
    vec3.lerp( color , vec3.fromValues(.5,.7,1.)  , vec3.fromValues(1,1,1) , t );
    return color;
  }
  else if (this.sky === 'image') {
    //get px and py
    let pixel = this.camera.getPixels(ray);

    const px = pixel[0];
    const py = pixel[1];

    const nx = this.background_image.width;
    const ny = this.background_image.height;
    //convert to i and j

    //let px = (i + 0.5) / nx;
    let i = (px * nx) -.5;
    //let py = (ny - j - 0.5) / ny;
    let j = (py * ny + .5 - ny) * -1;

    i = Math.round(i);
    j = Math.round(j);

    //use them to get one index for the giant pixel array of image data
    const index_x = i;
    const index_y = j * this.background_image.width;
    let index = index_x + index_y;
    index = index * 4 ;

    //grab the pixel
    const r = this.data[index]/255;
    const g = this.data[index +1]/255;
    const b = this.data[index + 2]/255;


    let color = vec3.fromValues(r, g, b);

    //and return it!
    return color;
  }
  else{
    alert('unknown sky ',this.sky);
  }
}

RayTracer.prototype.setPixel = function( image , x , y , r , g , b ) {
  /**
   * Sets the pixel color into the image data that is ultimately shown on the canvas.
   *
   * @param image - image data to write to
   * @param x,y - pixel coordinates within [0,0] x [canvas.width,canvas.height]
   * @param r,g,b - color to assign to pixel, each channel is within [0,1]
   * @returns none
   *
   * You do not need to change this function.
  **/
  let offset = (image.width * y + x) * 4;
  image.data[offset  ] = 255*Math.min(r,1.0);
  image.data[offset+1] = 255*Math.min(g,1.0);
  image.data[offset+2] = 255*Math.min(b,1.0);
  image.data[offset+3] = 255; // alpha: transparent [0-255] opaque
}

function Sphere(params) {
  // represents a sphere object
  this.center   = params['center']; // center of the sphere (vec3)
  this.radius   = params['radius']; // radius of sphere (float)
  this.material = params['material']; // material used to shade the sphere (see 'Material' below)
  this.name     = params['name'] || 'sphere'; // a name to identify the sphere (useful for debugging) (string)
  this.color = params['color'];
  this.type = 'sphere';
}
Sphere.prototype.intersect = function(ray, tmin, tmax){
  let temp = vec3.create();
  //ray point, not camera.eye, to generalize it
  vec3.sub(temp, ray.point, this.center);

  let B = vec3.dot(ray.direction, temp);

  let C = vec3.dot(temp, temp) - (this.radius * this.radius);

  let discriminant = (B*B) - C;


  if (discriminant < 0){

    return undefined;
  }
  let t1 = -B - Math.sqrt(discriminant);
  let t2 = -B + Math.sqrt(discriminant);

  let t;

  let strikes = 1; //Initializes strikes to 1
  //you'll see why in a second
  if (t1 > tmin && t1 < tmax){
    t = t1;
  }
  else{
    strikes ++;//Strike two!
  }
  if(t2 > tmin && t2 < tmax && t2 < t1){
    t = t2;
  }
  else{
    strikes ++; //Strike three!!!
  }
  //andd.......
  if (strikes >= 3){
    //you're out!
    return undefined;
  }

  //calculate intersect point
  //let intersect = ray.point + ray.direction(t);
  let intersect = vec3.create();
  vec3.scale(intersect, ray.direction, t);
  vec3.add(intersect, intersect, ray.point);

  //calculate normal at intersect point
  //xyz coordinates translated from origin
  let n = vec3.create();
  let translate = vec3.create();
  const zero = vec3.fromValues(0, 0 ,0);
  vec3.sub(translate, zero, this.center);
  vec3.add(n, intersect, translate);
  vec3.normalize(n, n);

  //return hit object
  hit = {'t':t, 'color': this.color, 'material': this.material, 'intersection': intersect, 'normal':n, 'name':this.name};
  return hit;



}

function Light(params) {
  // describes a point light source, storing the location of the light
  // as well as ambient, diffuse and specular components of the light
  this.location = params.location; // location of
  this.color    = params.color || vec3.fromValues(1,1,1); // default to white (vec3)
  // you might also want to save some La, Ld, Ls and/or compute these from 'this.color'
  this.Ld = vec3.fromValues(1, 1, 1); //for now
  this.Ls = vec3.fromValues(1, 1, 1); //for now
}

function Material( params ) {
  // represents a generic material class
  this.type  = params.type; // diffuse, reflective, refractive (string)
  this.shine = params.shine; // phong exponent (float)
  this.color = params.color || vec3.fromValues(0.5,0.5,0.5); // default to gray color (vec3)
  this.strength = params.strength || .3;//strength of reflective or refractive properties

  // you might also want to save some ka, kd, ks and/or compute these from 'this.color'
  ka = vec3.create();
  vec3.scale(ka, this.color, .4)
  this.ka = ka;
  this.kd = this.color; //for now
  this.ks = vec3.fromValues(1, 1, 1); //for now

//this is really arbitrary
}
Material.prototype.shade = function(ray, light, hit){

  let l = vec3.create();

  //get l direction to light
  vec3.sub(l, light.location, hit.intersection);
  vec3.normalize(l, l);
  //calculate the diffuse component
  const ndotl = vec3.dot(hit.normal, l);
  let diffuse = vec3.create();
  vec3.multiply(diffuse, hit.material.kd, light.Ld);
  const scalar = Math.max(ndotl, 0);
  vec3.scale(diffuse, diffuse, scalar );

  //calculate the specular component
  let h = vec3.create();
  let v = vec3.create();
  //why did I have to flip this I shouldn't have had to flip this
  vec3.scale(v, ray.direction, -1);
  //normalize v?
  vec3.normalize(v, v);
  vec3.add(h, v, l);
  vec3.normalize(h, h);

  //normalize n even though it is already normalized
  vec3.normalize(hit.normal, hit.normal);
  const ndoth = Math.abs(vec3.dot(hit.normal, h));

  //handler for undefined shininess
  if(hit.material.shine === undefined){
    hit.material.shine = 20; //idk hardcoded for now

  }
  //just another scalar
  const component = Math.max(Math.pow(ndoth, hit.material.shine), 0);

  let spec = vec3.create();
  vec3.multiply(spec, hit.material.ks, light.Ls);

  vec3.scale(spec, spec, component);

  //add them together
  let color = vec3.create();
  vec3.add(color, diffuse, spec);

  return color;



}
Material.prototype.scatter = function(ray, hit){
  //for now, just reflect
  if (this.type === "reflective"){
    let n = hit.normal;
    let v = ray.direction;
    //vec3.scale(v, v, -1);
    //not sure these normalizations are necessary but I figured they can't hurt
    vec3.normalize(v, v);
    vec3.normalize(n, n);
    let ray_dir = this.reflect(n, v);
    //see above
    vec3.normalize(ray_dir, ray_dir);
    let ray_pt = hit.intersection;
    let new_ray = new Ray({'point': ray_pt, 'direction': ray_dir});
    //here we might get away with the abnormal direction just being the direction * 10 or so (approximation!)
    let ab = vec3.create();
    vec3.scale(ab, ray_dir, 10);
    new_ray.abnormal_direction = ab;
    return new_ray;
  }
  else if (this.type === "refractive"){
    let n = hit.normal;
    let v = ray.direction;
    //vec3.scale(v, v, -1);
    //not sure these normalizations are necessary but I figured they can't hurt
    vec3.normalize(v, v);
    vec3.normalize(n, n);
    let ray_dir = this.refract(n, v, 1.5);
    //see above
    //vec3.normalize(ray_dir, ray_dir);
    let ray_pt = hit.intersection;
    let new_ray = new Ray({'point': ray_pt, 'direction': ray_dir});
    //here we might get away with the abnormal direction just being the direction * 10 or so
    let ab = vec3.create();
    vec3.scale(ab, ray_dir, 10);
    new_ray.abnormal_direction = ab;
    return new_ray;
  }

}
Material.prototype.reflect = function(n, v){
  //reflect!
  let r = vec3.create();
  vec3.scale( r , n , -2.0*vec3.dot(v,n) );
  vec3.add( r , r , v );

  return r;
}
Material.prototype.refract = function(n, v, eta){
  //refract
  let dt = vec3.dot(v,n);
  let n1_over_n2 = 0;

  if(dt <= 0){
  //this means we are on the way in
    n1_over_n2 = 1/eta;
  }
  else{
    //we are on way out
    n1_over_n2 = eta/1;
    //flip the normal
    vec3.scale(n, n, -1);
    //recalculate dt
    dt = vec3.dot(v,n);
  }


  let discriminant = 1.0 - n1_over_n2*n1_over_n2*( 1.0 - dt*dt );
  if (discriminant < 0.0) {
    // total internal reflection
    return this.reflect(v,n);
  }

  let r1 = vec3.create();
  vec3.scaleAndAdd( r1 , v , n , -dt );
  vec3.scale( r1 , r1 , n1_over_n2 );

  let r2 = vec3.create();
  vec3.scale( r2 , n , Math.sqrt(discriminant) );

  let r = vec3.create();
  vec3.subtract(r,r1,r2);

  return r;
}

function Camera(params) {
  this.eye = params.eye; //vec 3 camera position
  this.center = params.center;
  this.up = params.up;
  this.fov = params.fov;
  this.aspect = params.aspect;

}
Camera.prototype.setUp = function(){
  let gaze = vec3.create();
  //gaze = lookat - eye
  vec3.sub(gaze, this.center, this.eye);
  this.gaze = gaze;

  //w = -gaze (normalized)
  let w = vec3.create();
  let zero = vec3.fromValues(0, 0, 0);
  vec3.normalize(w, gaze);
  vec3.sub(w, zero, w);

  //u = up x w
  let u = vec3.create();
  vec3.cross(u, this.up, w);
  vec3.normalize(u, u);

  //v = w x u
  let v = vec3.create();
  vec3.cross(v, w, u);

  //compute change of basis
  //transformation to camera space
  let B = mat3.create();
  //translation matrix
  let T = mat3.create();

  //remember the wacky way which glm sets up matrices
  //here we are setting first column = u, second to v, third to w
  for(let i = 0; i < 3; i++){
    B[i] = u[i];
    B[i+3] = v[i];
    B[i+6] = w[i];
    T[6+i] = this.eye[i]; //eye of i. or is it i of eye? haha
  }
  this.basis = B;
  Binv = mat3.create();
  mat3.invert(Binv, B);
  this.inv_basis = Binv;
  this.translation = T;



  //calculate d, h, w
  let dvec = vec3.create();
  vec3.sub(dvec, this.center, this.eye);
  let d = vec3.length(dvec) + this.eye[2];

  let h = 2*d * Math.tan(this.fov/2) + this.eye[1];

  let w2 = this.aspect * h + this.eye[0];

  this.d = d;
  this.h = h;
  this.w = w2;

}

Camera.prototype.getRay = function(px, py){
   //expects px, py floats

  //pu = –w/2 + px*w
  let pu = -this.w / 2 + px * this.w;

  //pv = –h/2 + py*h
  let pv = -this.h / 2 + py * this.h;

  let pw = -this.d;


  let q = vec3.fromValues(pu, pv, pw);


  let direction = vec3.create();
  vec3.transformMat3(direction, q, this.basis);
  //save this so that we can undo this transformation
  //for getPixels/ calculating background image
  let abnormal_direction = vec3.create();
  vec3.copy(abnormal_direction, direction); //get it? not normal
  //console.log(abnormal_direction);
  vec3.normalize(direction, direction);
  let pt = this.eye;
  ray = new Ray ({'point': pt, 'direction':direction })
  ray.abnormal_direction = abnormal_direction;
  this.getPixels(ray);
  return ray;


}
Camera.prototype.getPixels = function(ray){
   //expects a ray object
   //with property abnormal_direction;

  let point = vec3.create();

  vec3.transformMat3(point, ray.abnormal_direction, this.inv_basis);
  //console.log(this.basis);

  let pu = point[0];

  //pu = –w/2 + px*w
  //px * w = pu + w/2
  //px = (pu + w/2)/w
  let px = (pu + (this.w/2))/this.w;
  //console.log(px);

  let pv = point[1];
  //pv = –h/2 + py*h
  //py * h = pv + h/2
  //py = (pv + h/2)/h
  let py = (pv + (this.h/2))/this.h;

  let pixel = vec2.fromValues(px, py);

  return pixel;


}
function Ray(params) {
  // this.px = params.px;
  // this.py = params.py;
  this.point = params.point;
  this.direction = params.direction;



}

function Mesh(params) {
  this.triangles = params.triangles;
  this.vertices = params.vertices;
  this.material = params.material;
  this.type = 'mesh'; //for distinguishing in objects

}
Mesh.prototype.setUp = function(){
  //deals with the raw triangle data
  //converts to a nicer format of triangles perhaps
  let triangle_list = [];
  for (let i = 0; i < this.triangles.length; i+=3){
    let ind1 = this.triangles[i];
    let ind2 = this.triangles[i+1];
    let ind3 = this.triangles[i+2];



    let pt1 = vec3.fromValues(this.vertices[3*ind1], this.vertices[3*ind1+1], this.vertices[3*ind1+2]);

    let pt2 = vec3.fromValues(this.vertices[3*ind2], this.vertices[3*ind2+1], this.vertices[3*ind2+2]);

    let pt3 = vec3.fromValues(this.vertices[3*ind3], this.vertices[3*ind3+1], this.vertices[3*ind3+2]);
    let tri = new Triangle({'p1': pt1, 'p2':pt2, 'p3':pt3, 'material': this.material});
    //console.log(tri);
    triangle_list.push(tri);
  }
  return triangle_list;
  //returns list of triangle objects
}
function Triangle(params) {
  this.p1 = params.p1;
  this.p2 = params.p2;
  this.p3 = params.p3;
  this.material = params.material;
  this.type = 'triangle';
}
Triangle.prototype.intersect = function(ray, tmin, tmax) {
  //given a ray, does it intersect with a triangle

  //deference p1 , p2, p3 to make life easier
  const p1 = this.p1;
  const p2 = this.p2;
  const p3 = this.p3;

  //we have a system of three equations to find the weights
  //create the vectors that the coefficient matrix is based on
  let first = vec3.create();
  vec3.sub(first, p1, p3);

  let second = vec3.create();
  vec3.sub(second, p2, p3);


  let third = vec3.create();
  vec3.scale(third, ray.direction, -1);
  //create the coefficient matrix:
  let A = mat3.create();
  A[0] = first[0]; A[3] = second[0]; A[6] = third[0];
  A[1] = first[1]; A[4] = second[1]; A[7] = third[1];
  A[2] = first[2]; A[5] = second[2]; A[8] = third[2];

  let Ainv = mat3.create();
  mat3.invert( Ainv , A );


  let b = vec3.create();
  vec3.sub(b, ray.point, p3);

  let x = vec3.create();
  vec3.transformMat3(x, b, Ainv);


  let alpha = x[0];
  let beta = x[1];
  let t = x[2]
  let gamma = 1- alpha - beta;
  let intersect = false;
  //check if the intersect conditions are met
  if (alpha >= 0 && alpha <= 1){
    if (beta >= 0 && beta <= 1){
      if (gamma >= 0 && gamma <=1){
        if (t < tmax && t > tmin){
        intersect = true;
        }
      }

    }
  }
  if (intersect === false){
    return undefined;

  }
  //compute normal
  //p2 -p1
  let u = vec3.create();
  vec3.sub(u, p2, p1);


  //p3 - p1
  let v = vec3.create();
  vec3.sub(v, p3, p1);

  //n = u cross v
  let normal = vec3.create();
  vec3.cross(normal, u, v);

  //once we have t we can find intersection point
  //intersect = ray.point + ray.direction(t);
  let intersection = vec3.create();
  vec3.scale(intersection, ray.direction, t);
  vec3.add(intersection, intersection, ray.point);


  hit = {'t':t, 'material': this.material, 'intersection': intersection, 'normal':normal};

  return hit;

}
