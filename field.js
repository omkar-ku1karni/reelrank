/*
 * The ribbon field from Omkar Kulkarni's portfolio, shared by every project page.
 * One fixed WebGL canvas behind the page; falls back to a still CSS ground,
 * and freezes under prefers-reduced-motion.
 */
(function () {
  var canvas = document.createElement('canvas');
  canvas.className = 'field';
  canvas.setAttribute('aria-hidden', 'true');
  document.body.prepend(canvas);
  var gl = canvas.getContext('webgl', { antialias: false, alpha: false, powerPreference: 'low-power' });
  var fallback = function () {
    canvas.style.background =
      'radial-gradient(60% 50% at 82% 30%, rgb(106 108 246 / .28), transparent 70%),' +
      'radial-gradient(40% 35% at 95% 10%, rgb(255 106 61 / .22), transparent 70%), #0b0d1a';
  };
  if (!gl) return fallback();

  var VERT = 'attribute vec2 aPos;void main(){gl_Position=vec4(aPos,0.,1.);}';
  var FRAG = [
    'precision highp float;uniform vec2 uRes;uniform float uTime;uniform vec2 uMouse;uniform float uMouseOn;uniform float uScroll;',
    'float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}',
    'float noise(vec2 p){vec2 i=floor(p),f=fract(p);vec2 u=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1,0)),u.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),u.x),u.y);}',
    'float fbm(vec2 p){float v=0.,a=.5;for(int i=0;i<3;i++){v+=a*noise(p);p=p*2.03+vec2(1.7,9.2);a*=.5;}return v;}',
    'void main(){vec2 uv=gl_FragCoord.xy/uRes;vec2 p=(gl_FragCoord.xy-.5*uRes)/uRes.y;',
    'vec2 m=(uMouse-.5*uRes)/uRes.y;float dm=length(p-m);p.y+=(p.y-m.y)*.25*exp(-dm*dm*10.)*uMouseOn;',
    'float t=uTime*.12;p.y+=uScroll*.45;',
    'vec3 lilac=vec3(.416,.424,.965);vec3 ember=vec3(1.,.416,.239);vec3 base=vec3(.043,.051,.102);vec3 col=base;',
    'for(int i=0;i<3;i++){float fi=float(i);float x=p.x;',
    'float c=-.12+fi*.24+.42*x+.16*sin(x*(1.6+fi*.35)+t*(.7+fi*.15)+fi*2.1)+.14*(fbm(vec2(x*1.1+fi*3.7,t*.4))-.5);',
    'float d=p.y-c;float w=.085+.04*sin(x*1.9+fi*1.3+t*.6);float env=exp(-(d*d)/(w*w));',
    'float st=pow(1.-abs(sin((d/w)*7.+fbm(vec2(x*2.4,fi+t*.3))*3.)),5.);float core=exp(-(d*d)/(w*w*.06));',
    'float gr=step(.965,hash(floor(gl_FragCoord.xy)+fi*17.))*env;float al=smoothstep(-.2,.95,x+.25*fi-.1);',
    'col+=mix(lilac,ember,al)*(.55+.25*fi)*(env*st*.42+core*.55+gr*.55+env*.06);}',
    'float mask=smoothstep(.45,.85,uv.x)*.8;float calm=1.-.75*clamp(uScroll,0.,1.);',
    'col=base+(col-base)*mask*calm;col+=(hash(gl_FragCoord.xy+fract(uTime)*91.7)-.5)*.03;gl_FragColor=vec4(col,1.);}'
  ].join('');

  function sh(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  var prog;
  try {
    prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    gl.useProgram(prog);
  } catch (e) {
    return fallback();
  }
  var buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  var aPos = gl.getAttribLocation(prog, 'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
  var U = function (n) { return gl.getUniformLocation(prog, n); };
  var uRes = U('uRes'), uTime = U('uTime'), uMouse = U('uMouse'), uMouseOn = U('uMouseOn'), uScroll = U('uScroll');

  var SCALE = 0.5;
  var calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches || navigator.webdriver;
  var mouse = { x: 0, y: 0, tx: 0, ty: 0, on: 0, ton: 0 };
  var scroll = 0;
  function resize() {
    canvas.width = Math.max(1, Math.round(innerWidth * SCALE));
    canvas.height = Math.max(1, Math.round(innerHeight * SCALE));
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform2f(uRes, canvas.width, canvas.height);
  }
  resize();
  addEventListener('resize', function () { resize(); if (calm) frame(0); });
  addEventListener('pointermove', function (e) {
    if (e.pointerType !== 'mouse') return;
    mouse.tx = e.clientX * SCALE; mouse.ty = (innerHeight - e.clientY) * SCALE; mouse.ton = 1;
  }, { passive: true });

  var start = performance.now(), raf = 0;
  function frame(now) {
    mouse.x += (mouse.tx - mouse.x) * .06; mouse.y += (mouse.ty - mouse.y) * .06; mouse.on += (mouse.ton - mouse.on) * .04;
    scroll += (scrollY / Math.max(1, innerHeight * 2.5) - scroll) * .08;
    gl.uniform1f(uTime, calm ? 40 : (now - start) / 1000);
    gl.uniform2f(uMouse, mouse.x, mouse.y);
    gl.uniform1f(uMouseOn, calm ? 0 : mouse.on);
    gl.uniform1f(uScroll, scroll);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (!calm) raf = requestAnimationFrame(frame);
  }
  if (calm) {
    frame(0);
    addEventListener('scroll', function () { requestAnimationFrame(frame); }, { passive: true });
  } else {
    raf = requestAnimationFrame(frame);
    document.addEventListener('visibilitychange', function () {
      cancelAnimationFrame(raf);
      if (!document.hidden) raf = requestAnimationFrame(frame);
    });
  }
})();
