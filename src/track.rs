use crate::global::Point;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;


pub fn gen_track(grain: usize, res: usize, size: (Point, Point), width: f32, rng: &mut ChaCha8Rng) -> (Vec<Point>, Vec<Point>) {

    let mut track: Vec<Point> = vec![];

    for _ in 0..grain {
        track.push(Point::new(rng.gen_range(size.0.x..size.1.x), rng.gen_range(size.0.y..size.1.y)));
    }
    track = calculate_convex_hull(&track);
    track = chaikin_corner_cutting(&track, res);
    let track2 = dualise(&track, width);
    (track, track2)
}




fn calculate_convex_hull(points: &Vec<Point>) -> Vec<Point> {
    //There must be at least 3 points
    if points.len() < 3 { return points.clone(); }

    let mut hull = vec![];

    //Find the left most point in the polygon
    let (left_most_idx, _) = points.iter()
        .enumerate()
        .min_by(|lhs, rhs| lhs.1.x.partial_cmp(&rhs.1.x).unwrap())
        .expect("No left most point");

    let mut p = left_most_idx;
    let mut q: usize;

    loop {
        //The left most point must be part of the hull
        hull.push(points[p].clone());

        q = (p + 1) % points.len();

        for i in 0..points.len() {
            if orientation(&points[p], &points[i], &points[q]) == 2 {q = i;}
        }

        p = q;
        //Break from loop once we reach the first point again
        if p == left_most_idx { break; }
    }
    hull
}

//Calculate orientation for 3 points
//0 -> Straight line
//1 -> Clockwise
//2 -> Counterclockwise
fn orientation(p: &Point, q: &Point, r: &Point) -> usize {
    let val = (q.y - p.y) * (r.x - q.x) -
        (q.x - p.x) * (r.y - q.y);

    if val == 0. { return 0 };
    if val > 0. { 1 } else { 2 }
}

fn chaikin_corner_cutting(points: &[Point], iterations: usize) -> Vec<Point> {
    let mut points = points.to_owned();
    for _ in 0..iterations {
        let mut result = vec![];
        for i in 0..points.len() {
            let p0 = points[i].clone();
            let p1 = points[(i + 1) % points.len()].clone();
            let q0 = Point::new((p0.x * 2.0 + p1.x) / 3.0, (p0.y * 2.0 + p1.y) / 3.0);
            let q1 = Point::new((p0.x + p1.x * 2.0) / 3.0, (p0.y + p1.y * 2.0) / 3.0);
            result.push(q0);
            result.push(q1);
        }
        points = result;
    }
    points
}

fn dualise(points: &[Point], distance: f32) -> Vec<Point> {
let mut track: Vec<Point> = vec![];

    for i in 1..points.len()+1 {
        let angle = (points[(i+1)%points.len()].y - points[i-1].y).atan2(points[(i+1)%points.len()].x - points[i-1].x);
        track.push(Point::new(points[i%points.len()].x+distance*angle.sin(), points[i%points.len()].y-distance*angle.cos()));
    }
    track
}
