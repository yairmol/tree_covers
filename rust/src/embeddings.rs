pub fn embedding_distortion<T: PartialOrd + Copy>(
    points: &Vec<T>,
    d1: &mut impl FnMut(&T, &T) -> f32,
    d2: &mut impl FnMut(&T, &T) -> f32
) -> f32 {
    let mut distortion = 1.;
  
    for x in points {
        for y in points {
            if !(y > x) {continue;}
            let cur_distortion = d2(x, y) / d1(x, y);
            distortion = if distortion > cur_distortion {distortion} else {cur_distortion};
      }
    }
    distortion
  }