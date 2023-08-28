use std::f32::consts::PI;
use std::time::Instant;
use macroquad::time::get_fps;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use bincode::{serialize, deserialize};
//use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::prelude::*;

use macroquad::{miniquad::conf::Platform, window::*, shapes::*, text::draw_text};
use macroquad::prelude::{Color, GRAY, BLACK, DARKGRAY, is_key_pressed, KeyCode, is_key_down};


use rayon::prelude::*;
//use tensor_bitch_cpu::NN;
use rusty_neat::NN;

mod global;
use global::{Point, point_in_polygon, closest_index, move_perp, get_angle, average_distance};
mod track;
use track::gen_track;
mod car;
use car::{raywrap, Car};



pub const SEED: u64 = 1;  // seed for number generators 
pub const MAP_GRAIN: usize = 24;  // kind a smoothness of track
pub const MAP_RES: usize = 3;  // amount of chaikin's corner cutting iterations
pub const WINDOW_SIZE: (u32, u32) = (1920, 1080);//(1600, 900);
pub const TRACK_WIDTH: f32 = 110.0;
pub const ENTITIES_AMOUNT: usize = 8000;  // amount of cars in one generation
pub const RAY_AMOUNT: usize = 16;  // amount of rays on entities ( 16 best )
pub const MUT_STRENGTH: f64 = 0.2;  // mutation deviation strength
pub const GEN_LEN: f32 = 20.0;  // max initial time for each generation
pub const TEXT_COOLDOWN: f32 = 2.0;  // text fade after that time
pub const DATA_PATH: &str = "dat2.bin";
pub const STATIC_DT: bool = true;


#[macroquad::main(conf)]
async fn main() {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    // generate track
    let (mut track, mut track2) = gen_track(
        MAP_GRAIN, 
        MAP_RES, 
        (Point::new(TRACK_WIDTH, TRACK_WIDTH), Point::new(WINDOW_SIZE.0 as f32 - TRACK_WIDTH, WINDOW_SIZE.1 as f32 - TRACK_WIDTH)), 
        TRACK_WIDTH,
        &mut rng
    );
    let mut track_l = track.len();
    let dst_mod = 1750.0 / average_distance(&track);

    let mut entities: Vec<(Car, NN, bool)> = vec![];
    let mut entities_old: Vec<(Car, NN, bool)>;

    for _ in 0..ENTITIES_AMOUNT {
        entities.push(
            (
                (
                Car::new(
                    vec![Point::new(-13.0, -20.0), Point::new(13.0, -20.0), Point::new(13.0, 20.0), Point::new(-13.0, 20.0)], 
                    move_perp(&track[track_l/2-1], &track[track_l/2], &track[track_l/2+1], TRACK_WIDTH/2.0), 
                    get_angle(&track[track_l/2-1], &track[track_l/2+1]) + (PI/2.0) * (rng.gen_range(0..1) * 2 - 1) as f32,
                    1.0, 
                    0.9
                )
            ), 
                //NN::new(&[RAY_AMOUNT + 2, 6, 12, 2]), 
                NN::new(RAY_AMOUNT + 2, 2), 
                true
            )
        );
    }
    let mut best: usize = 0;  // index to best entity

    let mut alive_sum: usize = entities.len();

    let mut clock = Instant::now();
    let mut dt: f32;
    let mut dt_clock = Instant::now();
    let mut clock_save = Instant::now();
    let mut clock_read = Instant::now();
    let mut generation: usize = 0;
    
    loop {
        // move to next gen
        if clock.elapsed().as_secs_f32() > GEN_LEN + generation as f32 * 3.0 || alive_sum < 1 || is_key_pressed(KeyCode::L) {
            clock = Instant::now();
            generation += 1;
            
            // generate track
            (track, track2) = gen_track(
                MAP_GRAIN, 
                MAP_RES, 
                (Point::new(TRACK_WIDTH, TRACK_WIDTH), Point::new(WINDOW_SIZE.0 as f32 - TRACK_WIDTH, WINDOW_SIZE.1 as f32 - TRACK_WIDTH)), 
                TRACK_WIDTH,
                &mut rng
            );

            track_l = track.len();

            // save best entities
            entities_old = entities.clone();
            entities = vec![];

            if is_key_down(KeyCode::L) {
                let save = load(DATA_PATH);
                generation = save.1;
                entities_old.iter_mut().for_each(|e|{
                    *e = save.0.clone();
                });
                clock_read = Instant::now();
                best = 0;
            }

            let dir = if generation % 2 == 0 {-1}
            else {1};

            // create new entities
            for i in 0..ENTITIES_AMOUNT {

                // randomise weight sets from last gen, if any
                if !entities_old.is_empty() {entities.push( entities_old[weighted_random_select(&entities_old, &mut rng)].clone() );}
                else {entities.push( entities_old[best].clone() );}

                //if i != 0 {entities.last_mut().unwrap().1.mutate(MUT_STRENGTH / (generation as f64 / 3.0), &mut rng);}
                if i != 0 {entities.last_mut().unwrap().1.mutate();}
                
                entities.last_mut().unwrap().0.reset(
                    move_perp(&track[track_l/2-1], &track[track_l/2], &track[track_l/2+1], TRACK_WIDTH/2.0), 
                    get_angle(&track[track_l/2-1], &track[track_l/2+1]) + (PI/2.0) * dir as f32
                );
                entities.last_mut().unwrap().2 = true;
            }
        }



        if STATIC_DT { dt = 0.03333; }
        else {dt = dt_clock.elapsed().as_secs_f32();}
        dt_clock = Instant::now();
        if is_key_pressed(KeyCode::Q) {return;}
        
        // update best
        alive_sum = 0;
        entities.iter().enumerate().filter(|(_, e)| e.2).for_each(|(i, e)|{
            if e.0.get_score() > entities[best].0.get_score() || !entities[best].2 {best = i;}
            alive_sum += 1;
        });
        if is_key_pressed(KeyCode::S) {save(&entities[best], generation, DATA_PATH); clock_save = Instant::now();}

        // update, time is consumed by raytracing and nn.forward
        entities.par_iter_mut().enumerate().filter(|(_, e)| e.2).for_each(|(_i, e)| {
            // raytrace distances
            let (rv, _rp) = raywrap(e.0.get_position(), *e.0.get_angle(), RAY_AMOUNT, &track, &track2);

            // assign points for distance travelled
            let id = closest_index(e.0.get_position(), &track) as i128 - closest_index(e.0.get_position_last(), &track) as i128;
            if id > -(track.len() as i128 / 2) && id < (track.len() as i128 / 2) {e.0.add_dst(id);}
            
            e.0.add_score(
                //id.abs() * 10 +   // 64 * x / rev
                e.0.get_velocity().length_project(e.0.get_angle()) as i128      // 100_000 / rev
            );
            
            let mut ins: Vec<f64> = vec![];
            ins.push(e.0.get_velocity().length() as f64);
            ins.push(*e.0.get_velocity_ang() as f64);
            rv.iter().for_each(|p|{
                ins.push(*p as f64);
            });

            //e.1.forward(ins.as_slice());

            //e.0.acc_forward(e.1.get_output()[0] as f32 * 100.0, dt);
            //e.0.acc_ang(e.1.get_output()[1] as f32 * 1.5, dt);
            let o = e.1.forward(ins.as_slice());
            
            e.0.acc_forward(o[0] as f32 * 100.0, dt);
            e.0.acc_ang(o[1] as f32 * 1.5, dt);

            e.0.update(dt);

            // die if
            e.0.points.iter().for_each(|p|{
                if 
                point_in_polygon(p, &track) 
                || !point_in_polygon(p, &track2) 
                || e.0.get_dst().abs() + 3 < (clock.elapsed().as_secs_f32() * dst_mod ) as i128
                {e.2 = false;}
            });

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

        // print only 200 alive entities, for performance reasons
        entities.iter().filter(|e| e.2).take(200).for_each(|e| {
            for i in 0..e.0.points.len() {
                draw_line(
                    e.0.points[i].x, e.0.points[i].y, 
                    e.0.points[(i+1)%e.0.points.len()].x, e.0.points[(i+1)%e.0.points.len()].y, 
                    4.0, 
                    Color::from_rgba(255, 0, 0, 192)
                );
            }
        });


        //draw_nn(&entities[best]);
        
        // fps
        draw_text(&("FPS: ".to_owned() + &(get_fps()).to_string()), WINDOW_SIZE.0 as f32 - 110.0, 20.0, 30.0, DARKGRAY);
        // gen number
        draw_text(&("GEN: ".to_owned() + &(generation).to_string()), 10.0, WINDOW_SIZE.1 as f32 - 10.0, 50.0, DARKGRAY);
        // time
        draw_text(&("Time: ".to_owned() + &(clock.elapsed().as_secs()).to_string()), 10.0, WINDOW_SIZE.1 as f32 - 50.0, 30.0, DARKGRAY);
        // alive number
        draw_text(&("SUM: ".to_owned() + &(alive_sum).to_string()), WINDOW_SIZE.0 as f32 - 200.0, WINDOW_SIZE.1 as f32 - 10.0, 50.0, DARKGRAY);

        // Show screens
        if clock_read.elapsed().as_secs_f32() > TEXT_COOLDOWN {draw_cooldown(clock, &("GENERATION: ".to_owned() + &generation.to_string()), 100.0);}
        if generation > 0 {
            draw_cooldown(clock_save, &("SAVED GEN: ".to_owned() + &generation.to_string()), 200.0);
            draw_cooldown(clock_read, &("LOADED GEN: ".to_owned() + &generation.to_string()), 200.0);
        }
        // std::thread::sleep(std::time::Duration::from_millis(4));
        next_frame().await;
    }
}





fn save(entity: &(Car, NN, bool), gen: usize, path: &str) {
    let data = (entity.clone(), gen);
    
    let encoded: Vec<u8> = serialize(&data).unwrap();
    
    // open file and write whole Vec<u8>
    let mut file = File::create(path).unwrap();
    file.write_all(&encoded).unwrap();
}

fn load(path: &str) -> ((Car, NN, bool), usize) {
    // convert readed Vec<u8> to plain nn
    let mut buffer = vec![];
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();
    let decoded: ((Car, NN, bool), usize) = deserialize(&buffer).unwrap();

    decoded
}

fn draw_cooldown(clock: Instant, text: &str, size: f32) {
    if clock.elapsed().as_secs_f32() < TEXT_COOLDOWN {
        draw_text(
            text,
            (WINDOW_SIZE.0/2) as f32 - (text.len() as f32*size/5.0), WINDOW_SIZE.1 as f32 / (size / 92.0), 
            size, 
            Color::from_rgba(255, 255, 255, 255 - (((clock.elapsed().as_secs_f32())-TEXT_COOLDOWN+1.0) * 255.0).max(0.0) as u8)
        );
    }
}
//// draw best neural network
//fn draw_nn(entity: &(Car, NN, bool)) {
//    // println!("{}", entity.0.get_score());
//    // draw best entity
//    for i in 0..entity.0.points.len() {
//        draw_line(
//            entity.0.points[i].x, entity.0.points[i].y, 
//            entity.0.points[(i+1)%entity.0.points.len()].x, entity.0.points[(i+1)%entity.0.points.len()].y, 
//            4.0, 
//            Color::from_rgba(0, 255, 0, 192)
//        );
//    }
//
//    let w = entity.1.get_weights();
//
//    for (i, l) in w.iter().enumerate() {
//        for y1 in 0..l.len_of(ndarray::Axis(1)) {
//            for y0 in 0..l.len_of(ndarray::Axis(0)) {
//                let str = *l.get((y0, y1)).unwrap() as f32;
//                draw_line(
//                    (192 * i + 32) as f32, (32 * y1 + 32) as f32, 
//                    (192 * (i+1) + 32) as f32, (32 * y0 + 32) as f32, 
//                    4.0 * str.abs(), 
//                    Color::from_rgba((255.0 * str.max(0.0)) as u8, 0, (-255.0 * str.min(0.0)) as u8, 100)
//                )
//            }
//        }
//    }
//
//    let b = entity.1.get_biases();
//    let l = entity.1.get_layers();
//   
//    for (i, l) in l.iter().enumerate() {
//        for c in 0..l.len_of(ndarray::Axis(0)){
//            if i > 0 {
//                let str = *b[i-1].get(c).unwrap();
//                draw_circle((192 * i + 32) as f32, (32 * c + 32) as f32, 12.0, 
//                    Color::from_rgba((255.0 * str.max(0.0)) as u8, 0, (-255.0 * str.min(0.0)) as u8, 100)
//                );
//            } else {
//                draw_circle((192 * i + 32) as f32, (32 * c + 32) as f32, 12.0, Color::from_rgba(255, 255, 255, 100));
//            }
//            draw_circle_lines((192 * i + 32) as f32, (32 * c + 32) as f32, 12.0, 1.0, Color::from_rgba(0, 0, 0, 100));
//        }
//    }
//}

fn weighted_random_select(struct_vec: &Vec<(Car, NN, bool)>, rng: &mut ChaCha8Rng) -> usize {
    // Step 1: Create a vector of tuples
    let score_tuples = struct_vec.iter().enumerate().map(|(i, s)|{
        (i, s.0.get_score().pow(2))
    }).collect::<Vec<_>>();
    
    // Step 2: Find the total score
    let total_score: u128 = score_tuples.iter().map(|(_, score)| *score).sum();
    
    // Step 3: Generate a random number
    let random_num = rng.gen_range(0..total_score);
    
    // Step 4: Iterate through the tuple vector
    let mut current_score = 0;
    for (index, score) in score_tuples {
        current_score += score;
        if random_num < current_score {
            // Step 5: Return the Struct at the index
            return index;
        }
    }
    struct_vec.len() - 1
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
