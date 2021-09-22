var cat_filter = (() => {
  // This mostly comes from this repo: https://github.com/takuti/poisson-image-blending
  const blendImages = function(size, base_canvas, src_canvas, mask_canvas, result_canvas) {
    let base_ctx = base_canvas.getContext('2d');
    let src_ctx = src_canvas.getContext('2d');
    let mask_ctx = mask_canvas.getContext('2d');
    let result_ctx = result_canvas.getContext('2d');

    let base_pixels = base_ctx.getImageData(0, 0, size.width, size.height);
    let src_pixels = src_ctx.getImageData(0, 0, size.width, size.height);
    let mask_pixels = mask_ctx.getImageData(0, 0, size.width, size.height);
    let result_pixels = result_ctx.getImageData(0, 0, size.width, size.height);

    let dx, absx, previous_epsilon=1.0;
    let cnt=0;

    do {
      dx=0; absx=0;
      for (let y=1; y<size.height-1; y++) {
        for (let x=1; x<size.width-1; x++) {
          let p = (y*size.width+x)*4;

          if (mask_pixels.data[p+0]==0 && mask_pixels.data[p+1]==255 &&
              mask_pixels.data[p+2]==0 && mask_pixels.data[p+3]==255) {

            let p_offseted = p;

            // q is array of connected neighbors
            let q = [((y-1)*size.width+x)*4, ((y+1)*size.width+x)*4,
                      (y*size.width+(x-1))*4, (y*size.width+(x+1))*4];
            let num_neighbors = q.length;

            for (let rgb=0; rgb<3; rgb++) {
              let sum_fq = 0;
              let sum_vpq = 0;
              let sum_boundary = 0;

              for (let i=0; i<num_neighbors; i++) {
                let q_offseted = q[i];

                if (mask_pixels.data[q[i]+0]==0 && mask_pixels.data[q[i]+1]==255 &&
                    mask_pixels.data[q[i]+2]==0 && mask_pixels.data[q[i]+3]==255) {
                  sum_fq += result_pixels.data[q_offseted+rgb];
                } else {
                  sum_boundary += base_pixels.data[q_offseted+rgb];
                }

                sum_vpq += src_pixels.data[p+rgb]-src_pixels.data[q[i]+rgb];
              }

              let new_value = (sum_fq+sum_vpq+sum_boundary)/num_neighbors;
              dx += Math.abs(new_value-result_pixels.data[p_offseted+rgb]);
              absx += Math.abs(new_value);
              result_pixels.data[p_offseted+rgb] = new_value;
            }
          }
        }
      }
      cnt++;
      let epsilon = dx/absx;
      if (!epsilon || previous_epsilon-epsilon < 0.0001) break; // convergence
      else previous_epsilon = epsilon;

      if (cnt > 10) break;

    } while(true);

    result_ctx.putImageData(result_pixels, 0, 0);
  };

  const draw_face = function(canvas_full, box, size) {
    //const canvas = document.getElementById('face_canvas');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = size;
    canvas.height = size;

    const ratio = 1.0 * box.height / box.width;
    const width = ratio <= 1 ? size : 1.0 * size / ratio;
    const height = ratio >= 1 ? size : 1.0 * size * ratio;

    ctx.drawImage(canvas_full, box.x, box.y, box.width, box.height, 0, 0, width, height);
    const dim = { width: width, height: height };
    return { canvas: canvas, dim: dim };
  };

  const draw_filter = async function(model, canvas_face) {
    const ctx_face = canvas_face.getContext('2d');
    const imgData = ctx_face.getImageData(0, 0, canvas_face.width, canvas_face.height);
    let im = tf.browser.fromPixels(imgData, 3).toFloat();
    im = tf.div(tf.sub(tf.div(im, 255), 0.5), 0.5);

    const input = tf.transpose(im, [2, 0, 1]).expandDims(0);
    let output = await model.executeAsync(input);

    let res = tf.transpose(tf.gather(output, 0), [1, 2, 0])
    res = tf.mul(tf.add(res, 1), 0.5);

    //const canvas_filter = document.getElementById('filter_canvas');
    const canvas_filter = document.createElement('canvas');
    await tf.browser.toPixels(res, canvas_filter);

    const ctx_filter = canvas_filter.getContext('2d');
    ctx_filter.filter = 'blur(0.6px)';
    ctx_filter.drawImage(canvas_filter, 0, 0, canvas_filter.width, canvas_filter.height);

    return canvas_filter;
  };

  const get_model = async function(filter_path, shape, executeAsync) {
    const model = await tf.loadGraphModel(filter_path+'/model.json');
    const input = tf.zeros(shape);
    if (executeAsync) await model.executeAsync(input);
    else              await model.execute(input);
    return model;
  };

  const get_cat_face_box = async function(canvas) {
    const box_model = cat_box_model;

    const ctx = canvas.getContext('2d');
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let input = tf.browser.fromPixels(imgData, 3).toFloat();
    input = tf.div(input, 255).resizeBilinear([224, 224]).expandDims(0);
    const output = box_model.execute(input);

    const bb = tf.reshape(output, [2, 2]);
    const center = await tf.mean(bb, 0).data();

    const d = await bb.data();
    const face_size = Math.max(Math.abs(d[2] - d[0], d[3] - d[1]));
    const c = 0.8;
    const new_bb = [
      [ center[0] - face_size*c, center[1] - face_size*c ],
      [ center[0] + face_size*c, center[1] + face_size*c ]
    ];

    const ratio_w = canvas.width/224;
    const ratio_h = canvas.height/224;
    const clamp = (num, min, max) => Math.min(Math.max(num, min), max);
    const x1 = clamp(new_bb[0][0]*ratio_w, 0, canvas.width);
    const x2 = clamp(new_bb[1][0]*ratio_w, 0, canvas.width);
    const y1 = clamp(new_bb[0][1]*ratio_h, 0, canvas.height);
    const y2 = clamp(new_bb[1][1]*ratio_h, 0, canvas.height);
    return { x: x1, width: x2-x1, y: y1, height: y2-y1 };
  };

  const drawImage = function(ctx, image, x, y, w, h, degrees) {
    ctx.save();
    //ctx.translate(x+w/2, y+h/2);
    ctx.rotate(degrees*Math.PI/180.0);
    //ctx.translate(-x-w/2, -y-h/2);
    ctx.drawImage(image, x, y, w, h);
    ctx.restore();
  };

  const get_aligned_face = async function(img_canvas, landmarks, face_canvas) {
    const canvas_aligned = document.createElement('canvas');
    canvas_aligned.width = img_canvas.width;
    canvas_aligned.height = img_canvas.height;

    const ctx = canvas_aligned.getContext('2d');
    const face_ctx = face_canvas.getContext('2d');

    const lm = landmarks;

    const dim = Math.max(
      Math.abs(lm.right_eye.x - lm.left_eye.x)*2,
      Math.abs(lm.left_eye.y - lm.mouth.y)*2
    );

    const p1 = lm.left_eye;
    const p2 = lm.right_eye;
    const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x) * 180 / Math.PI;
    drawImage(ctx, img_canvas, 0, 0, img_canvas.width, img_canvas.height, -angle);

    const mean = pts => pts.reduce((a, b) => a+b)/pts.length;
    const mean_pt = {
      x: mean([ lm.left_eye.x, lm.right_eye.x ]),
      y: mean([ lm.left_eye.y, lm.right_eye.y ])
    };
    const dist = (p1, p2) => Math.sqrt(Math.pow(p1.x-p2.x, 2) + Math.pow(p1.y-p2.y, 2));
    const width = dist(lm.left_eye, lm.right_eye)*3.5;
    const height = width;
    face_ctx.drawImage(canvas_aligned, mean_pt.x-width/2, mean_pt.y-height/2, width, height, 0, 0, face_canvas.width, face_canvas.height);

    return {
      width: width, height: height, angle: angle,
      pt: { x: mean_pt.x-width/2, y: mean_pt.y-height/2 }
    };
  };

  const get_cat_landmarks = async function(canvas, bounding_box, face_dim) {
    const landmarks_model = cat_landmarks_model;

    const ctx = canvas.getContext('2d');
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let input = tf.browser.fromPixels(imgData, 3).toFloat();
    input = tf.div(input, 255).resizeBilinear([224, 224]).expandDims(0);
    const landmarks = await landmarks_model.execute(input).data();

    const bb = bounding_box;
    const ratio_w = (bb.width/face_dim.width) * canvas.width/224;
    const ratio_h = (bb.height/face_dim.height) * canvas.height/224;
    return {
      left_eye:  { x: bb.x + landmarks[0]*ratio_w, y: bb.y + landmarks[1]*ratio_h },
      right_eye: { x: bb.x + landmarks[2]*ratio_w, y: bb.y + landmarks[3]*ratio_h },
      mouth:     { x: bb.x + landmarks[4]*ratio_w, y: bb.y + landmarks[5]*ratio_h }
    };
  };

  const replace_face = function(img_canvas, face_canvas, filter_canvas, box) {
    //const mask_canvas = document.getElementById('mask_canvas');
    const mask_canvas = document.createElement('canvas');
    const mask_ctx = mask_canvas.getContext('2d');
    mask_canvas.width = filter_size;
    mask_canvas.height = filter_size;

    const pad = 5;
    mask_ctx.drawImage(filter_canvas, 0, 0, mask_canvas.width, mask_canvas.height);
    mask_ctx.beginPath();
    mask_ctx.arc(mask_canvas.width/2, mask_canvas.height/2, mask_canvas.width/2-pad, 0, 2*Math.PI);
    mask_ctx.fillStyle = 'rgba(0, 255, 0, 1.0)';
    mask_ctx.fill();

    //const blend_canvas = document.getElementById('blend_canvas');
    const blend_canvas = document.createElement('canvas');
    const blend_ctx = blend_canvas.getContext('2d');
    blend_canvas.width = filter_size;
    blend_canvas.height = filter_size;

    blend_ctx.drawImage(face_canvas, 0, 0, blend_canvas.width, blend_canvas.height);

    const size = { width: mask_canvas.width, height: mask_canvas.height };
    blendImages(size, face_canvas, filter_canvas, mask_canvas, blend_canvas);

    const img_ctx = img_canvas.getContext('2d');
    drawImage(img_ctx, blend_canvas, box.pt.x, box.pt.y, box.width, box.height, box.angle);
  };

  const apply_filter = async function(src, model) {
    const img_canvas = await image(src);

    let bb = await get_cat_face_box(img_canvas);
    let res = await draw_face(img_canvas, bb, filter_size);
    let face_canvas   = res.canvas;
    const face_dim    = res.dim;
    const landmarks = await get_cat_landmarks(face_canvas, bb, face_dim);
    const box = await get_aligned_face(img_canvas, landmarks, face_canvas);

    const filter_canvas = await draw_filter(model, face_canvas);

    replace_face(img_canvas, face_canvas, filter_canvas, box);

    return img_canvas;
  };

  let filter_size = 256;

  let cat_box_model;
  let cat_landmarks_model;

  let has_done_setup = false;
  const setup = async function(config) {
    await tf.setBackend('wasm');
    await tf.enableProdMode();
    await tf.ENV.set('DEBUG', false);
    await tf.ready();

    if (config === undefined) config = {};
    if (!('bounding_box_model_path' in config))
      config['bounding_box_model_path'] = 'cat_box_uint8';

    if (!('landmarks_model_path' in config))
      config['landmarks_model_path'] = 'cat_landmarks_uint8';

    shape = [1, 224, 224, 3];
    cat_box_model       = await get_model(config['bounding_box_model_path'], shape, false);
    cat_landmarks_model = await get_model(config['landmarks_model_path'], shape, false);

    has_done_setup = true;
  };

  const load = async function(model_path, setup_config) {
    if (!has_done_setup) await setup(setup_config);

    let shape = [1, 3, filter_size, filter_size];
    return await get_model(model_path, shape, true);
  };

  const image = async function(url) {
    const imgSize = 1000;
    return new Promise((resolve) => {
      const img = new Image();
      img.addEventListener('load', () => {
        const ratio = 1.0 * img.height / img.width;
        img.width = ratio <= 1 ? imgSize : 1.0 * imgSize / ratio;
        img.height = ratio >= 1 ? imgSize : 1.0 * imgSize * ratio;

        //const canvas = document.getElementById('img_canvas');
        const canvas = document.createElement('canvas');
        canvas.height = img.height;
        canvas.width = img.width;
        const ctx = canvas.getContext('2d');
        if (ctx) ctx.drawImage(img, 0, 0, img.width, img.height);
        ctx.imageSmoothingQuality = 'high';

        resolve(canvas);
      });
      img.src = url;
    });
  };

  const apply = async function(img, model) {
    let canvas = await apply_filter(img.src, model);
    img.src = canvas.toDataURL();
  };

  return {
    setup: setup,
    load: load,
    apply: apply
  };
})();
