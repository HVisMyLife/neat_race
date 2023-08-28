use std::f32::consts::PI;

use serde::{Serialize, Deserialize};


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32
} 
    
impl Point {
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
    
    pub fn length(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
    pub fn length_project(&self, a: &f32) -> f32 {
        self.x * (a+PI/2.0).cos() + self.y * (a+PI/2.0).sin()
    }
} 

pub fn point_in_polygon(point: &Point, polygon: &Vec<Point>) -> bool {
    let mut intersections = 0;
    for i in 0..polygon.len() {
        let a = polygon[i].clone();
        let b = polygon[(i + 1) % polygon.len()].clone();
        if (a.y > point.y) != (b.y > point.y) &&
            point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x
        {
            intersections += 1;
        }
    }
    intersections % 2 == 1
}

pub fn distance(p1: &Point, p2: &Point) -> f32 {
    ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt()
}

pub fn average_distance(points: &Vec<Point>) -> f32 {
    let mut distance_sum = 0.0;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let dx = points[i].x - points[j].x;
            let dy = points[i].y - points[j].y;
            let distance = (dx.powi(2) + dy.powi(2)).sqrt();
            distance_sum += distance;
        }
    }
    distance_sum / (points.len() * (points.len() - 1) / 2) as f32
}

pub fn closest_index(point: &Point, polygon: &[Point]) -> usize {
    let mut index: (usize, f32) = (0, f32::MAX);

    polygon.iter().enumerate().for_each(|(i, p)|{
        let dst = distance(point, p);
        if dst < index.1 {index.0 = i; index.1 = dst;}
    });

    index.0
}

pub fn get_angle(p1: &Point, p2: &Point) -> f32 {
    (p2.y - p1.y).atan2(p2.x - p1.x)
}

// move perpendicular to two points
pub fn move_perp(p1: &Point, p: &Point, p2: &Point, distance: f32) -> Point {
    let angle = get_angle(p1, p2);
    Point::new(p.x+distance*angle.sin(), p.y-distance*angle.cos())
}
