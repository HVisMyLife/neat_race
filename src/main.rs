use std::f32::consts::PI;
use std::time::Instant;
use std::usize;
use macroquad::time::get_fps;
use rand::prelude::*;

use macroquad::{miniquad::conf::Platform, window::*, shapes::*, text::draw_text};
use macroquad::prelude::{Color, GRAY, BLACK, DARKGRAY, is_key_pressed, KeyCode};


use rayon::prelude::*;
use rusty_neat::{NeatIntermittent, NN, ActFunc};

mod global;
use global::{Point, point_in_polygon, closest_index, move_perp, get_angle, average_distance};
mod track;
use track::gen_track;
mod car;
use car::{raywrap, Car};



pub const MAP_GRAIN: usize = 24;  // kind a smoothness of track
pub const MAP_RES: usize = 3;  // amount of chaikin's corner cutting iterations
pub const WINDOW_SIZE: (u32, u32) = (1920, 1080);//(1600, 900);
pub const TRACK_WIDTH: f32 = 110.0;
pub const ENTITIES_AMOUNT: usize = 2000;  // amount of cars in one generation
pub const RAY_AMOUNT: usize = 8;  // amount of rays on entities ( 16 best )
pub const GEN_LEN: f32 = 20.0;  // max initial time for each generation
pub const TEXT_COOLDOWN: f32 = 2.0;  // text fade after that time
pub const STATIC_DT: bool = true;
pub const RECURRENCE: bool = true;


#[macroquad::main(conf)]
async fn main() {
    let mut rng = rand::rng();
    
    // generate track
    let (mut track, mut track2) = gen_track(
        MAP_GRAIN, 
        MAP_RES, 
        (Point::new(TRACK_WIDTH, TRACK_WIDTH), Point::new(WINDOW_SIZE.0 as f32 - TRACK_WIDTH, WINDOW_SIZE.1 as f32 - TRACK_WIDTH)), 
        TRACK_WIDTH,
    );
    let mut track_l = track.len();
    let dst_mod = 1750.0 / average_distance(&track);

    let mut neat = NeatIntermittent::new( 
        &NN::new(RAY_AMOUNT + 2, 2, None, RECURRENCE, 0.75,
            ActFunc::SigmoidBipolar, &[ActFunc::SigmoidBipolar, ActFunc::SELU, ActFunc::HyperbolicTangent]), 
        ENTITIES_AMOUNT, 7 );
    neat.speciate();
    let mut cars = vec![];
    (0..neat.agents.len()).into_iter().for_each(|_| 
        cars.push(
            Car::new(
                vec![Point::new(-13.0, -20.0), Point::new(13.0, -20.0), Point::new(13.0, 20.0), Point::new(-13.0, 20.0)], 
                move_perp(&track[track_l/2-1], &track[track_l/2], &track[track_l/2+1], TRACK_WIDTH/2.0), 
                get_angle(&track[track_l/2-1], &track[track_l/2+1]) + (PI/2.0) * (rng.random_range(0..1) * 2 - 1) as f32,
                1.0, 
                0.9)
        )
    );

    let mut alive_sum: usize = ENTITIES_AMOUNT;

    let mut clock = Instant::now();
    let mut dt: f32;
    let mut dt_clock = Instant::now();
    let mut generation: usize = 0;
    let mut fta = FrameTimeAnalyzer::new(32);
    
    loop {
        //println!("{:?}", neat.agents[0]);
        // move to next gen
        if clock.elapsed().as_secs_f32() > GEN_LEN + generation as f32 * 3.0 || alive_sum < 1 {
            clock = Instant::now();
            generation += 1;
            
            // generate track
            (track, track2) = gen_track(
                MAP_GRAIN, 
                MAP_RES, 
                (Point::new(TRACK_WIDTH, TRACK_WIDTH), Point::new(WINDOW_SIZE.0 as f32 - TRACK_WIDTH, WINDOW_SIZE.1 as f32 - TRACK_WIDTH)), 
                TRACK_WIDTH,
            );

            track_l = track.len();
            neat.agents.iter_mut().for_each(|a| a.fitness = a.fitness.sqrt().sqrt() );

            neat.next_gen();
            neat.mutate(None);
            neat.speciate();
            //while neat.species_table.len() != neat.species_amount {neat.speciate();}

            cars.clear();

            (0..neat.agents.len()).into_iter().for_each(|_| 
                cars.push(
                    Car::new(
                        vec![Point::new(-13.0, -20.0), Point::new(13.0, -20.0), Point::new(13.0, 20.0), Point::new(-13.0, 20.0)], 
                        move_perp(&track[track_l/2-1], &track[track_l/2], &track[track_l/2+1], TRACK_WIDTH/2.0), 
                        get_angle(&track[track_l/2-1], &track[track_l/2+1]) + (PI/2.0) * (rng.random_range(0..1) * 2 - 1) as f32,
                        1.0, 
                        0.9)
                )
            );
        }

        dt = if STATIC_DT { 0.03333 } else { dt_clock.elapsed().as_secs_f32() };
        dt_clock = Instant::now();
        if is_key_pressed(KeyCode::Q) {return;}
        
        alive_sum = cars.iter().filter(|c| c.alive ).count();


        let mut ins: Vec<Vec<f32>> = vec![vec![]; cars.len()];
        cars.par_iter_mut().zip_eq(ins.par_iter_mut()).for_each(|(c, i)| {
            if c.alive {
            let (mut rv, _rp) = raywrap(c.get_position(), *c.get_angle(), RAY_AMOUNT, &track, &track2);

            // track checkpoints travelled 
            let id = closest_index(c.get_position(), &track) as isize - closest_index(c.get_position_last(), &track) as isize;
            if id > -(track.len() as isize / 2) && id < (track.len() as isize / 2) {c.distance += id as isize;}
            // speed in forward direction
            c.agility += c.get_velocity().length_project(c.get_angle());

            i.push(c.get_velocity().length());
            i.push(*c.get_velocity_ang());
            i.append(&mut rv); }
        } );
        neat.forward(&ins);
        cars.par_iter_mut().zip_eq(neat.agents.par_iter_mut()).for_each(|(c,a)|{
            if c.alive {
            let o = a.get_outputs();
            c.acc_forward(o[0]*100., dt);
            c.acc_ang(o[1]*4., dt);
            c.update(dt);
            c.alive = !c.points.iter().any(|p|{ // death check 
                point_in_polygon(p, &track) ||
                !point_in_polygon(p, &track2) ||
                c.distance.abs() + 3 < ( clock.elapsed().as_secs_f32() * dst_mod ) as isize
            });
            a.active = c.alive;
            a.fitness = c.agility.max(0.001);}
        });

        // ----------------- DRAWING

        // clear background
        clear_background(GRAY);

        // draw tracks
        for i in 0..track.len() {
            draw_line(
                track[i].x, track[i].y, 
                track[(i+1)%track.len()].x, track[(i+1)%track.len()].y, 
                3.0, BLACK
            );
        }
        for i in 0..track2.len() {
            draw_line(
                track2[i].x, track2[i].y, 
                track2[(i+1)%track2.len()].x, track2[(i+1)%track2.len()].y, 
                3.0, BLACK
            );
        }

        // don't print all entities, for performance reasons
        let colors = neat.species_table.keys().cloned().collect::<Vec<usize>>();
        cars.iter().zip(neat.agents.iter()).filter(|(c,_)| c.alive).for_each(|(c,n)| {
            for i in 0..c.points.len() {
                draw_line(
                    c.points[i].x, c.points[i].y, 
                    c.points[(i+1)%c.points.len()].x, c.points[(i+1)%c.points.len()].y, 
                    4.0, 
                    contrasting_color(&colors, n.species)
                );
            }
        });
        //// print species leaders
        //let colors = neat.species_table.keys().cloned().collect::<Vec<usize>>();
        //colors.iter().for_each(|s| {
        //    let best = neat.agents.iter().enumerate().filter(|(_,a)| a.species == *s)
        //        .map(|(i,a)| (i, a.fitness) ).max_by(|a,b| a.partial_cmp(b).unwrap() ).unwrap();
        //    let c = &cars[best.0];
        //
        //    for i in 0..c.points.len() {
        //        draw_line(
        //            c.points[i].x, c.points[i].y, 
        //            c.points[(i+1)%c.points.len()].x, c.points[(i+1)%c.points.len()].y, 
        //            4.0, 
        //            contrasting_color(&colors, *s)
        //        );
        //    }
        //} );

        //draw_nn(&entities[best]);
        
        // fps
        fta.add_frame_time(get_fps() as f32);
        draw_text(&("FPS: ".to_owned() + &(fta.smooth_frame_time()).to_string()), WINDOW_SIZE.0 as f32 - 110.0, 20.0, 30.0, DARKGRAY);
        // gen number
        draw_text(&("GEN: ".to_owned() + &(generation).to_string()), 10.0, WINDOW_SIZE.1 as f32 - 10.0, 50.0, DARKGRAY);
        // time
        draw_text(&("Time: ".to_owned() + &(clock.elapsed().as_secs()).to_string()), 10.0, WINDOW_SIZE.1 as f32 - 50.0, 30.0, DARKGRAY);
        // alive number
        draw_text(&("SUM: ".to_owned() + &(alive_sum).to_string()), WINDOW_SIZE.0 as f32 - 200.0, WINDOW_SIZE.1 as f32 - 10.0, 50.0, DARKGRAY);

        // Show screens
        //if clock_read.elapsed().as_secs_f32() > TEXT_COOLDOWN {draw_cooldown(clock, &("GENERATION: ".to_owned() + &generation.to_string()), 100.0);}
        //if generation > 0 {
        //    draw_cooldown(clock_save, &("SAVED GEN: ".to_owned() + &generation.to_string()), 200.0);
        //    draw_cooldown(clock_read, &("LOADED GEN: ".to_owned() + &generation.to_string()), 200.0);
        //}
        // std::thread::sleep(std::time::Duration::from_millis(4));
        next_frame().await;
    }
}


fn contrasting_color(slice: &[usize], element: usize) -> Color {
    let len = slice.len();
    if len == 0 {
        return Color::from_rgba(255,255,255,255); // Or some default color
    }

    let index = slice.iter().position(|&x| x == element).unwrap_or(0);

    let hue = (index as f32 / len as f32) * 360.0;

    //  saturation and value are constants, 
    // make sure to have high Value for contrast.

    hsv_to_rgb(hue, 0.8, 0.9)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color {
    let h_i = (h / 60.0) as i32;
    let f = h / 60.0 - h_i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    let (r, g, b) = match h_i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        5 => (v, p, q),
        _ => (0.0, 0.0, 0.0), // Should not happen, h is always in range [0, 360)
    };

    Color::new(r, g, b, 1.0) //alpha = 255
}


fn _draw_cooldown(clock: Instant, text: &str, size: f32) {
    if clock.elapsed().as_secs_f32() < TEXT_COOLDOWN {
        draw_text(
            text,
            (WINDOW_SIZE.0/2) as f32 - (text.len() as f32*size/5.0), WINDOW_SIZE.1 as f32 / (size / 92.0), 
            size, 
            Color::from_rgba(255, 255, 255, 255 - (((clock.elapsed().as_secs_f32())-TEXT_COOLDOWN+1.0) * 255.0).max(0.0) as u8)
        );
    }
}

pub struct FrameTimeAnalyzer {
    frame: Vec<f32>,
    s_time: f32,
}

impl FrameTimeAnalyzer {
    pub fn new(length: usize) -> Self {
        FrameTimeAnalyzer {
            frame: vec![0.; length],
            s_time: 0.,
        }
    }

    pub fn add_frame_time(&mut self, time: f32) {
        self.frame.pop();
        self.frame.insert(0, time);
    }

    pub fn smooth_frame_time(&mut self) -> &f32 {
        self.s_time = self.frame.iter().sum::<f32>() / (self.frame.len() as f32);
        &self.s_time
    }
}

fn conf() -> Conf {
    let p = Platform {
        linux_backend: macroquad::miniquad::conf::LinuxBackend::X11WithWaylandFallback,
        ..Default::default()
    };
    Conf {
        window_title: String::from("Macroquad"),
        window_width: WINDOW_SIZE.0 as i32,
        window_height: WINDOW_SIZE.1 as i32,
        fullscreen: false,
        sample_count: 16,
        platform: p,
        ..Default::default()
    }
}
