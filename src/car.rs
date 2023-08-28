use crate::global::Point;
use std::f32::consts::PI;
use serde::{Serialize, Deserialize};


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Car {
    pub points: Vec<Point>,
    points_relative: Vec<Point>,
    position: Point,
    position_last: Point,
    velocity: Point,
    angle: f32,
    velocity_ang: f32,
    mass: f32,
    friction: f32,
    score: u128,
    distance: i128
}

impl Car {
    pub fn new(points_relative: Vec<Point>, position: Point, angle: f32, mass: f32, friction: f32) -> Self { 
        Self { 
            points: points_relative.clone(), points_relative, 
            position: position.clone(), position_last: position, velocity: Point::new(0.0, 0.0), 
            angle, velocity_ang: 0.0, 
            mass, friction, score: 100, distance: 0
        } 
    }

    pub fn update(&mut self, dt: f32) {
        self.position_last = self.position.clone();

        self.velocity.x += self.velocity.x * -self.friction / self.mass * dt;
        self.velocity.y += self.velocity.y * -self.friction / self.mass * dt;
        self.velocity_ang += self.velocity_ang * -self.friction / self.mass * dt;

        self.position.x += self.velocity.x * dt;
        self.position.y += self.velocity.y * dt;
        self.angle += self.velocity_ang * dt;

        self.points.iter_mut().zip(self.points_relative.iter()).for_each(|(g, l)|{
            g.x = self.position.x + l.x * self.angle.cos() + l.y * self.angle.sin();
            g.y = self.position.y + l.x * self.angle.sin() - l.y * self.angle.cos();
        });
    }

    pub fn acc_forward(&mut self, acc: f32, dt: f32) {
        self.velocity.x += acc * (self.angle+PI/2.0).cos() / self.mass * dt;
        self.velocity.y += acc * (self.angle+PI/2.0).sin() / self.mass * dt;
    }

    pub fn acc_ang(&mut self, acc: f32, dt: f32) {
        self.velocity_ang += acc / self.mass * dt;
    }

    pub fn reset(&mut self, pos: Point, angle: f32) {
        self.position = pos.clone();
        self.position_last = pos;
        self.angle = angle;
        self.velocity_ang = 0.0;
        self.velocity = Point::new(0.0, 0.0);
        self.score = 100;
        self.distance = 0;
    }

    pub fn add_score(&mut self, s: i128) {
        if (self.score as i128 + s) >= 0 {self.score = (self.score as i128 + s) as u128;}
    }

    pub fn add_dst(&mut self, s: i128) {
        self.distance += s;
    }

    pub fn get_angle(&self) -> &f32 {
        &self.angle
    }

    pub fn get_position(&self) -> &Point {
        &self.position
    }
    pub fn get_position_last(&self) -> &Point {
        &self.position_last
    }

    pub fn get_score(&self) -> &u128 {
        &self.score
    }

    pub fn get_dst(&self) -> &i128 {
        &self.distance
    }

    pub fn get_velocity(&self) -> &Point {
        &self.velocity
    }

    pub fn get_velocity_ang(&self) -> &f32 {
        &self.velocity_ang
    }

    //pub fn get_points(&self) -> &Vec<Point> {
    //    &self.points
    //}

}


fn raycast(point: &Point, angle: f32, polygon: &Vec<Point>) -> Option<Point> {
    let mut min_t = std::f32::MAX;
    let mut intersection: Option<Point> = None;

    let x = point.x;
    let y = point.y;
    let dx = angle.cos();
    let dy = angle.sin();

    for i in 0..polygon.len() {
        let j = (i + 1) % polygon.len();
        let ax = polygon[i].x;
        let ay = polygon[i].y;
        let bx = polygon[j].x;
        let by = polygon[j].y;

        let t = ((ay - y) * (bx - ax) - (ax - x) * (by - ay)) / (dy * (bx - ax) - dx * (by - ay));

        if t >= 0.0 && (dx * t + x) >= ax.min(bx) && (dx * t + x) <= ax.max(bx) && (dy * t + y) >= ay.min(by) && (dy * t + y) <= ay.max(by) && t < min_t {
            min_t = t;
            intersection = Some(Point::new(dx * t + x, dy * t + y));
        }
    }

    intersection
}

pub fn raywrap(point: &Point, angle: f32, amount: usize, track: &Vec<Point>, track2: &Vec<Point>) -> (Vec<f32>, Vec<Point>) {
    let mut dsts: Vec<f32> = vec![];
    let mut pp: Vec<Point> = vec![];

    for i in 0..amount {
        
        let ray1 = raycast(point, (2.0*PI/amount as f32)*i as f32+PI/2.0+angle, track);
        let ray2 = raycast(point, (2.0*PI/amount as f32)*i as f32+PI/2.0+angle, track2);

        if let (Some(ray1), Some(ray2)) = (ray1.clone(), ray2.clone()) {
            let d1 = ((ray1.x - point.x).powi(2)+(ray1.y - point.y).powi(2)).sqrt();
            let d2 = ((ray2.x - point.x).powi(2)+(ray2.y - point.y).powi(2)).sqrt();
            if d1 < d2 {dsts.push(d1);pp.push(ray1);}
            else {dsts.push(d2);pp.push(ray2);}
        }
        else if let Some(ray1) = ray1 {
            let d = ((ray1.x - point.x).powi(2)+(ray1.y - point.y).powi(2)).sqrt();
            dsts.push(d);
            pp.push(ray1);
        }
        else if let Some(ray2) = ray2 {
            let d = ((ray2.x - point.x).powi(2)+(ray2.y - point.y).powi(2)).sqrt();
            dsts.push(d);
            pp.push(ray2);
        }
        else {dsts.push(f32::MAX);pp.push(Point::new(-100.0, -100.0));}

    }
    (dsts, pp)
}
