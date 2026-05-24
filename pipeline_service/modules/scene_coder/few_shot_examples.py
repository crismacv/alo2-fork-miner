from __future__ import annotations


FEW_SHOT_EXAMPLES = """\
## Worked examples — study the patterns, then write your own module

These miniature exemplars demonstrate the idioms you should use for common
shapes. They are NOT a library to call — they are reference patterns.
Adapt them to whatever the reference image shows.

### Example 1 — Wooden chair (4 radial legs, seat, backrest)

Reference summary:
> A simple wooden chair with four straight cylindrical legs, a flat square
> seat, and a tall vertical-slat backrest. Walnut wood throughout.

```javascript
export default function generate(THREE) {
  // Materials — single shared wood material for every part keeps the
  // chair coherent and saves draw calls.
  const woodMat = new THREE.MeshStandardMaterial({
    color: 0x8b6f47,
    metalness: 0.0,
    roughness: 0.6,
  });

  const root = new THREE.Group();

  // Seat — flat square box, sits at mid-height.
  const seatGeom = new THREE.BoxGeometry(0.45, 0.04, 0.45);
  const seat = new THREE.Mesh(seatGeom, woodMat);
  seat.position.y = 0.40;
  root.add(seat);

  // Legs — 4 cylindrical legs, radial symmetric. Use one geometry +
  // four meshes so the model is robust to "wrong_count" critique.
  const legGeom = new THREE.CylinderGeometry(0.022, 0.022, 0.40, 16);
  const legPositions = [
    [ 0.20, 0.20,  0.20],
    [-0.20, 0.20,  0.20],
    [ 0.20, 0.20, -0.20],
    [-0.20, 0.20, -0.20],
  ];
  for (const [x, y, z] of legPositions) {
    const leg = new THREE.Mesh(legGeom, woodMat);
    leg.position.set(x, y, z);
    root.add(leg);
  }

  // Backrest — tall flat plate at the back of the seat.
  const backrestGeom = new THREE.BoxGeometry(0.45, 0.45, 0.025);
  const backrest = new THREE.Mesh(backrestGeom, woodMat);
  backrest.position.set(0, 0.65, -0.21);
  root.add(backrest);

  fitToUnitCube(THREE, root);
  return root;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
```

Key idioms:
- Single shared material for visually-uniform objects.
- Radial-symmetric placement via explicit position list (4 corners) —
  cleaner than computing angles unless count is large.
- Backrest is a thin Z-axis box, not a tall vertical plate; orientation
  matters for rendering.
- `fitToUnitCube` with `0.95 / maxDim` is mandatory.

### Furniture pattern — Upholstered loveseat / sofa

Reference summary:
> A two-seat sofa with soft cushions, rolled arms, visible seams/piping,
> button tufting, and separate legs or wood/metal frame pieces.

Pattern to reuse:

```javascript
const root = new THREE.Group();
const fabricMat = new THREE.MeshStandardMaterial({
  color: 0x7c3fb2,
  metalness: 0.0,
  roughness: 0.92,
});
const seamMat = new THREE.MeshStandardMaterial({
  color: 0x3f1f65,
  metalness: 0.0,
  roughness: 0.95,
});
const woodMat = new THREE.MeshStandardMaterial({
  color: 0x8a4f2c,
  metalness: 0.0,
  roughness: 0.58,
});

function addBox(name, w, h, d, mat, x, y, z) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  mesh.position.set(x, y, z);
  root.add(mesh);
  return mesh;
}

function addPipingX(x, y, z, length, mat) {
  const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, length, 10), mat);
  pipe.rotation.z = Math.PI / 2;
  pipe.position.set(x, y, z);
  root.add(pipe);
  return pipe;
}

function addPipingY(x, y, z, length, mat) {
  const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, length, 8), mat);
  pipe.position.set(x, y, z);
  root.add(pipe);
  return pipe;
}

function addRollArm(side) {
  const x = side * 0.55;
  addBox("arm_side", 0.11, 0.30, 0.58, fabricMat, x, 0.18, 0.02);
  const roll = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.60, 20), fabricMat);
  roll.rotation.x = Math.PI / 2; // cylinder axis runs along depth Z
  roll.position.set(x, 0.36, 0.02);
  root.add(roll);
  const frontCap = new THREE.Mesh(new THREE.CylinderGeometry(0.078, 0.078, 0.014, 20), seamMat);
  frontCap.rotation.x = Math.PI / 2;
  frontCap.position.set(x, 0.36, 0.33);
  root.add(frontCap);
}

function addButton(x, y, z) {
  const button = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.010, 16), seamMat);
  button.rotation.x = Math.PI / 2; // button faces forward from the back cushion
  button.position.set(x, y, z);
  root.add(button);
  // Short crease marks around the button suggest tufting without custom shaders.
  addPipingX(x, y, z + 0.004, 0.055, seamMat);
  addPipingY(x, y, z + 0.005, 0.045, seamMat);
}

// Seat base and distinct two-seat cushion modules.
addBox("front_apron", 1.05, 0.11, 0.08, fabricMat, 0, 0.11, 0.31);
for (const sx of [-0.26, 0.26]) {
  addBox("seat_cushion", 0.49, 0.085, 0.52, fabricMat, sx, 0.24, 0.03);
  addPipingX(sx, 0.29, 0.30, 0.46, seamMat);
}
addPipingY(0, 0.25, 0.03, 0.10, seamMat); // center gap between cushions

// Two back cushions, taller than the seats, with seam and button grid.
for (const sx of [-0.26, 0.26]) {
  addBox("back_cushion", 0.50, 0.44, 0.07, fabricMat, sx, 0.47, -0.25);
  for (const bx of [sx - 0.13, sx + 0.13]) {
    for (const by of [0.42, 0.55]) addButton(bx, by, -0.205);
  }
}
addPipingY(0, 0.47, -0.205, 0.40, seamMat);

// Rolled arms, rear rail, and separate frame/legs.
addRollArm(-1);
addRollArm(1);
addBox("rear_frame", 1.18, 0.10, 0.06, woodMat, 0, 0.14, -0.30);
addBox("front_frame", 1.18, 0.055, 0.055, woodMat, 0, 0.06, 0.34);
for (const [x, z] of [[-0.48, 0.28], [0.48, 0.28], [-0.48, -0.26], [0.48, -0.26]]) {
  addBox("leg", 0.055, 0.16, 0.055, woodMat, x, -0.04, z);
}
```

Key idioms:
- Use separate modules for seats/backs; never collapse a loveseat into one
  monolithic cushion block.
- Add rounded cues with cylinders/bolsters/piping even when the core is a box.
- Buttons and tufting belong on the back/arms in a grid, with small crease
  marks; they are not random dots.
- Rolled arms need top cylinders, side support slabs, and front cap discs.
- Upholstery, wood, metal, seams, buttons, and legs use separate materials.

### Furniture pattern — Slatted chaise lounge

```javascript
const lounge = new THREE.Group();
const plankMat = new THREE.MeshStandardMaterial({ color: 0x28a99c, metalness: 0.0, roughness: 0.65 });
const railMat = new THREE.MeshStandardMaterial({ color: 0x1f7771, metalness: 0.0, roughness: 0.75 });

function addBox(name, w, h, d, mat, x, y, z) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  mesh.position.set(x, y, z);
  lounge.add(mesh);
  return mesh;
}

function addPlank(x, y, z, w, h, d, rotX) {
  const p = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), plankMat);
  p.position.set(x, y, z);
  p.rotation.x = rotX;
  lounge.add(p);
  return p;
}

// Separate planks follow a shallow seat then reclined back; visible gaps matter.
for (let i = 0; i < 7; i++) addPlank(0, 0.12, 0.24 - i * 0.075, 0.58, 0.035, 0.052, 0.0);
for (let i = 0; i < 8; i++) addPlank(0, 0.18 + i * 0.045, -0.28 - i * 0.055, 0.58, 0.035, 0.052, -0.75);
addBox("left_rail", 0.045, 0.055, 1.10, railMat, -0.33, 0.05, -0.12);
addBox("right_rail", 0.045, 0.055, 1.10, railMat, 0.33, 0.05, -0.12);
for (const [x, z, rx] of [[-0.25, 0.30, -0.35], [0.25, 0.30, -0.35], [-0.25, -0.45, 0.45], [0.25, -0.45, 0.45]]) {
  const leg = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.34, 0.055), railMat);
  leg.position.set(x, -0.10, z);
  leg.rotation.x = rx;
  lounge.add(leg);
}
```

Key idioms:
- Slats are individual planks with gaps, not a single solid ramp.
- Backrest planks share a recline angle; base planks stay nearly horizontal.
- Rails and angled legs sit under the planks and remain visually connected.

### Example 2 — Glass bottle (lathe profile, transmission glass)

Reference summary:
> A clear glass wine bottle with a bulbous body tapering to a long neck
> and a small lip at the top. Empty, transparent.

```javascript
export default function generate(THREE) {
  // Glass material — MeshPhysicalMaterial with transmission for
  // see-through behavior. metalness 0, low roughness.
  const glassMat = new THREE.MeshPhysicalMaterial({
    color: 0xddeedd,
    metalness: 0.0,
    roughness: 0.05,
    transmission: 0.95,
    ior: 1.5,
    transparent: true,
  });

  // Lathe profile — array of THREE.Vector2(radius, height) points
  // describing the silhouette from bottom to top. CRITICAL: must be
  // Vector2 instances, not [r, y] arrays — plain arrays produce NaN
  // vertices and an invisible mesh.
  const profile = [
    new THREE.Vector2(0.00, 0.00),  // closed bottom center
    new THREE.Vector2(0.18, 0.00),  // bottom edge
    new THREE.Vector2(0.18, 0.10),  // shoulder of body
    new THREE.Vector2(0.18, 0.45),  // body top (still wide)
    new THREE.Vector2(0.10, 0.55),  // body→neck transition
    new THREE.Vector2(0.05, 0.60),  // neck base
    new THREE.Vector2(0.05, 0.85),  // neck top
    new THREE.Vector2(0.06, 0.90),  // small lip flare
    new THREE.Vector2(0.00, 0.92),  // close top opening
  ];
  const bodyGeom = new THREE.LatheGeometry(profile, 32);
  const bottle = new THREE.Mesh(bodyGeom, glassMat);

  const root = new THREE.Group();
  root.add(bottle);

  fitToUnitCube(THREE, root);
  return root;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
```

Key idioms:
- `new THREE.Vector2(r, y)` for every lathe profile point — NEVER
  `[r, y]` plain arrays.
- Profile starts at bottom (low y) and goes up; first/last points should
  have radius 0 if you want a closed shell.
- Glass uses `MeshPhysicalMaterial` with `transmission` + `ior` + `transparent`,
  not MeshStandardMaterial.
- Even single-part objects need a wrapping `Group` for `fitToUnitCube`.

### Example 3 — SUV (body, cabin, wheels, roof rack, spare tire)

Reference summary:
> A boxy mid-size SUV with a high roofline, chunky side steps, a roof rack,
> and a spare tire mounted on the rear door. Tan body paint, dark rubber
> wheels with chrome hub caps, and tinted glass windows.

```javascript
export default function generate(THREE) {
  const group = new THREE.Group();

  // --- dimension constants (all in local units before fitToUnitCube) ---
  const VW = 0.62, VL = 0.92;
  const wheelR = 0.042;
  const wheelBot = -0.23;
  const wheelCY = wheelBot + wheelR;
  const bodyBot = -0.17;
  const belt = 0.005;
  const roofBot = 0.16, roofTop = 0.185;
  const rackY = 0.20;
  const tireThick = wheelR * 0.35;
  const torusR = wheelR - tireThick;

  // --- materials: one per distinct surface class ---
  const bodyMat   = new THREE.MeshStandardMaterial({ color: 0xC8B896, roughness: 0.6, metalness: 0.1 });
  const blackMat  = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.7, metalness: 0.05 });
  const darkMat   = new THREE.MeshStandardMaterial({ color: 0x1A1A1A, roughness: 0.8, metalness: 0.05 });
  const chromeMat = new THREE.MeshStandardMaterial({ color: 0xC0C0C0, roughness: 0.2, metalness: 0.6 });
  const glassMat  = new THREE.MeshPhysicalMaterial({
    color: 0x8899AA, roughness: 0.1, metalness: 0.0,
    transmission: 0.5, transparent: true, opacity: 0.6,
  });
  const lensMat   = new THREE.MeshStandardMaterial({
    color: 0xFFFFDD, roughness: 0.3, metalness: 0.2,
    emissive: 0xFFFFDD, emissiveIntensity: 0.15,
  });
  const rackMat   = new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.5, metalness: 0.3 });
  const tireMat   = new THREE.MeshStandardMaterial({ color: 0x1A1A1A, roughness: 0.9, metalness: 0.0 });
  const hubMat    = new THREE.MeshStandardMaterial({ color: 0x3A3A3A, roughness: 0.4, metalness: 0.6 });
  const tailMat   = new THREE.MeshStandardMaterial({
    color: 0xCC2222, roughness: 0.3, metalness: 0.1,
    emissive: 0xCC2222, emissiveIntensity: 0.1,
  });

  // Helper — avoids repeating new THREE.Mesh(BoxGeometry...) boilerplate.
  function addBox(w, h, d, mat, x, y, z) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    group.add(m);
    return m;
  }

  // Helper — thin structural tube between two Vector3 points.
  function addTube(p1, p2, r, mat) {
    const m = new THREE.Mesh(
      new THREE.TubeGeometry(new THREE.LineCurve3(p1, p2), 1, r, 6, false),
      mat
    );
    group.add(m);
  }

  // --- body ---
  const lbW = VW * 0.80, lbH = belt - bodyBot, lbL = VL * 0.88;
  addBox(lbW, lbH, lbL, bodyMat, 0, bodyBot + lbH / 2, 0);

  const cabW = VW * 0.74, cabH = roofBot - belt, cabL = VL * 0.58, cabZ = -VL * 0.08;
  addBox(cabW, cabH, cabL, bodyMat, 0, belt + cabH / 2, cabZ);

  const hoodL = VL * 0.20;
  addBox(lbW * 0.92, 0.012, hoodL, bodyMat, 0, belt, cabZ + cabL / 2 + hoodL / 2 + 0.005);

  addBox(VW * 0.70, roofTop - roofBot, cabL * 0.96, blackMat, 0, roofBot + (roofTop - roofBot) / 2, cabZ);

  // --- roof rack: outer frame + cross-bars + corner uprights via addTube ---
  const rkW = VW * 0.55, rkL = VL * 0.42, tubR = 0.004;
  const corners = [
    new THREE.Vector3(-rkW / 2, rackY, cabZ - rkL / 2),
    new THREE.Vector3( rkW / 2, rackY, cabZ - rkL / 2),
    new THREE.Vector3( rkW / 2, rackY, cabZ + rkL / 2),
    new THREE.Vector3(-rkW / 2, rackY, cabZ + rkL / 2),
  ];
  for (let i = 0; i < 4; i++) addTube(corners[i], corners[(i + 1) % 4], tubR, rackMat);
  for (let ci = 1; ci <= 3; ci++) {
    const cz = corners[0].z + (corners[3].z - corners[0].z) * (ci / 4);
    addTube(new THREE.Vector3(-rkW / 2, rackY, cz), new THREE.Vector3(rkW / 2, rackY, cz), tubR, rackMat);
  }
  for (const c of corners) addTube(c, new THREE.Vector3(c.x, roofTop + 0.002, c.z), tubR, rackMat);

  // --- glass ---
  const wsH = cabH * 0.78;
  addBox(VW * 0.64, wsH, 0.005, glassMat, 0, belt + cabH * 0.12 + wsH / 2, cabZ + cabL / 2 + 0.003);
  const rwH = wsH * 0.72;
  addBox(VW * 0.52, rwH, 0.005, glassMat, 0, belt + cabH * 0.16 + rwH / 2, cabZ - cabL / 2 - 0.003);

  // Side windows: two per side, iterated with ±1 pattern.
  const swH = cabH * 0.55, swY = belt + cabH * 0.22 + swH / 2;
  for (const side of [-1, 1]) {
    const sx = side * (cabW / 2 + 0.003);
    const swFrontL = cabL * 0.30, swRearL = cabL * 0.25, swBase = cabZ + cabL / 2 - cabL * 0.06;
    addBox(0.005, swH,          swFrontL, glassMat, sx, swY, swBase - swFrontL / 2);
    addBox(0.005, swH * 0.92,   swRearL,  glassMat, sx, swY, swBase - swFrontL - cabL * 0.05 - swRearL / 2);
  }

  // --- front grille + chrome slats ---
  const grW = VW * 0.34, grH = 0.055, grZ = lbL / 2 + 0.005, grY = bodyBot + lbH * 0.52;
  addBox(grW, grH, 0.012, darkMat, 0, grY, grZ);
  for (let si = 0; si < 5; si++) {
    addBox(grW * 0.84, grH * 0.08, 0.016, chromeMat, 0, grY - grH / 2 + grH * (si + 0.5) / 5, grZ + 0.004);
  }

  // --- headlights: CylinderGeometry rotated 90° to face forward ---
  const hlR = VW * 0.038;
  for (const hs of [-1, 1]) {
    const hx = hs * (grW / 2 + hlR + 0.018), hy = grY + 0.005;
    const rim = new THREE.Mesh(new THREE.CylinderGeometry(hlR + 0.005, hlR + 0.005, 0.008, 16), chromeMat);
    rim.rotation.x = Math.PI / 2;
    rim.position.set(hx, hy, grZ);
    group.add(rim);
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(hlR, hlR, 0.012, 16), lensMat);
    lens.rotation.x = Math.PI / 2;
    lens.position.set(hx, hy, grZ + 0.002);
    group.add(lens);
  }

  // --- bumpers + fenders ---
  const bmpW = VW * 0.84, bmpH = 0.025, bmpD = 0.032;
  addBox(bmpW, bmpH, bmpD, blackMat, 0, bodyBot + bmpH / 2,  lbL / 2 + bmpD / 2);
  addBox(bmpW, bmpH, bmpD, blackMat, 0, bodyBot + bmpH / 2, -lbL / 2 - bmpD / 2);
  for (const fs of [-1, 1]) {
    addBox(0.012, lbH * 0.22, lbL * 0.9, blackMat, fs * (lbW / 2 + 0.005), bodyBot + lbH * 0.11, 0);
  }

  // --- wheels: TorusGeometry (tire) + CylinderGeometry (hub + cap),
  //     all rotated Math.PI/2 around Z so they face the X-axis. ---
  const wFZ =  VL * 0.30, wRZ = -VL * 0.30, wInX = lbW / 2;
  for (const [wx, wz] of [[-wInX, wFZ], [wInX, wFZ], [-wInX, wRZ], [wInX, wRZ]]) {
    const wy = wheelCY;
    const tire = new THREE.Mesh(new THREE.TorusGeometry(torusR, tireThick, 10, 24), tireMat);
    tire.rotation.z = Math.PI / 2;
    tire.position.set(wx, wy, wz);
    group.add(tire);

    const hub = new THREE.Mesh(new THREE.CylinderGeometry(wheelR * 0.42, wheelR * 0.42, 0.015, 12), hubMat);
    hub.rotation.z = Math.PI / 2;
    hub.position.set(wx, wy, wz);
    group.add(hub);

    const cap = new THREE.Mesh(new THREE.CylinderGeometry(wheelR * 0.12, wheelR * 0.12, 0.018, 8), chromeMat);
    cap.rotation.z = Math.PI / 2;
    cap.position.set(wx, wy, wz);
    group.add(cap);

    // Wheel arch
    const sideDir = wx > 0 ? 1 : -1;
    addBox(0.016, wheelR * 2.3, wheelR * 2.5, blackMat, wx + sideDir * 0.014, wy + wheelR * 0.35, wz);
  }

  // --- tail lights ---
  const tlW = 0.022, tlH = 0.032;
  for (const ts of [-1, 1]) {
    addBox(tlW, tlH, 0.008, tailMat, ts * (lbW / 2 - tlW * 0.6), bodyBot + lbH * 0.55, -lbL / 2 - 0.003);
  }

  // --- spare tire on rear door ---
  const spareThick = tireThick * 0.8;
  const spareTorusR = wheelR * 0.95 - spareThick;
  const spareZ = -lbL / 2 - bmpD - spareThick - 0.008;
  const spareY = bodyBot + lbH * 0.5;
  const spareTire = new THREE.Mesh(new THREE.TorusGeometry(spareTorusR, spareThick, 8, 20), tireMat);
  spareTire.rotation.x = Math.PI / 2;
  spareTire.position.set(0, spareY, spareZ);
  group.add(spareTire);
  const spareHub = new THREE.Mesh(new THREE.CylinderGeometry(wheelR * 0.35, wheelR * 0.35, 0.012, 10), hubMat);
  spareHub.rotation.x = Math.PI / 2;
  spareHub.position.set(0, spareY, spareZ);
  group.add(spareHub);

  // --- side steps + mirrors ---
  for (const ss of [-1, 1]) {
    addBox(0.028, 0.007, VL * 0.42, blackMat, ss * (lbW / 2 + 0.012), bodyBot + 0.008, 0);
    const mx = ss * (cabW / 2 + 0.018), mY = belt + cabH * 0.55, mZ = cabZ + cabL / 2 - cabL * 0.02;
    addBox(0.005, 0.018, 0.022, blackMat, mx, mY, mZ);
    addBox(0.003, 0.014, 0.018, glassMat, mx + ss * 0.003, mY, mZ);
  }

  fitToUnitCube(THREE, group);
  return group;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
```

Key idioms:
- Multiple materials (one per surface class: body, glass, chrome, rubber, emissive
  tail/lens) — never one global material for a multi-surface object.
- `addBox` / `addTube` helpers eliminate repeated `new THREE.Mesh(...)` for
  symmetric parts — extract helpers whenever the same pattern appears 4+ times.
- Wheels: `TorusGeometry` (tire ring) + `CylinderGeometry` (hub disc),
  both with `rotation.z = Math.PI/2` so they face the X-axis, not Y-up.
- Roof rack rails and cross-bars use `TubeGeometry` with `LineCurve3` for
  thin structural lines — not BoxGeometry.
- Symmetric pairs (wheels, windows, fenders, mirrors, steps) use
  `for (const side of [-1, 1])` so count is explicit and easy to verify.
- Spare tyre on rear door faces forward → `rotation.x = Math.PI/2` (Y-axis
  wheel), unlike the road wheels which use `rotation.z = Math.PI/2`.
- `fitToUnitCube` is still mandatory even for large multi-part assemblies.

### Example 3b — Multi-subject scene with BUILDER pattern (coffee cup ON a saucer)

Reference summary:
> A ceramic coffee cup sitting on top of a round flat saucer. TWO distinct
> objects; the saucer extends past the cup base on all sides. This is the
> canonical "X on Y" multi-subject case — model each subject as its own
> builder function, then COMPOSE them in `generate`.

Pattern to reuse (memorize this STRUCTURE — it generalizes to all
multi-subject scenes; only the geometry inside each builder changes):

```javascript
// One small builder per inventoried subject. Self-contained groups,
// own local coordinates, NEVER call fitToUnitCube here.
function buildSaucer(THREE) {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({
    color: 0xf2efe6, metalness: 0.0, roughness: 0.4,
  });
  const saucerR = 0.55;
  const thick  = 0.04;
  const base = new THREE.Mesh(
    new THREE.CylinderGeometry(saucerR, saucerR * 0.92, thick, 48),
    mat,
  );
  base.position.y = thick / 2;
  g.add(base);
  // Slight inner well where the cup foot will rest:
  const well = new THREE.Mesh(
    new THREE.CylinderGeometry(0.22, 0.22, 0.008, 32),
    mat,
  );
  well.position.y = thick + 0.004;
  g.add(well);
  g.userData.topY = thick + 0.008;  // explicit attachment surface
  return g;
}

function buildCup(THREE) {
  const g = new THREE.Group();
  const ceramic = new THREE.MeshStandardMaterial({
    color: 0xf2efe6, metalness: 0.0, roughness: 0.4,
  });
  const coffee = new THREE.MeshStandardMaterial({
    color: 0x3a2716, metalness: 0.0, roughness: 0.6,
  });
  const r = 0.2;
  const h = 0.32;
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(r, r * 0.88, h, 48, 1, true),
    ceramic,
  );
  body.position.y = h / 2;
  g.add(body);
  const bottom = new THREE.Mesh(
    new THREE.CylinderGeometry(r * 0.88, r * 0.88, 0.01, 32),
    ceramic,
  );
  bottom.position.y = 0.005;
  g.add(bottom);
  const surface = new THREE.Mesh(
    new THREE.CylinderGeometry(r * 0.96, r * 0.96, 0.005, 32),
    coffee,
  );
  surface.position.y = h - 0.02;
  g.add(surface);
  // Handle (torus arc on +X side, opening toward viewer):
  const handle = new THREE.Mesh(
    new THREE.TorusGeometry(0.085, 0.02, 12, 32, Math.PI),
    ceramic,
  );
  handle.position.set(r + 0.02, h / 2, 0);
  handle.rotation.y = Math.PI / 2;
  g.add(handle);
  return g;
}

export default function generate(THREE) {
  const root = new THREE.Group();
  // inventory: saucer, cup (n = 2)
  // layout: cup sits on top of the saucer's central well (X on Y)

  const saucer = buildSaucer(THREE);
  const cup    = buildCup(THREE);
  cup.position.y = saucer.userData.topY;  // explicit attachment, no float
  root.add(saucer);
  root.add(cup);

  fitToUnitCube(THREE, root);
  return root;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
```

Pattern notes:
- One builder function per inventoried subject (`buildSaucer`, `buildCup`).
  Builders are independent, work in their own local coordinates, and
  return finished Groups.
- `generate` is just inventory comment + builder calls + positioning +
  one `fitToUnitCube`. No mesh logic lives directly in `generate`.
- Attachment via `saucer.userData.topY` exposed by the saucer builder.
  No magic constants in `generate`; no air gap; no intersection.
- This same structure scales: 5 subjects → 5 builders + 5 calls.
  Repeated subjects (4 chair legs, 5 floating berries) call one builder
  in a loop and set `.position.set(...)` per instance.

### Example 3c — Container with internal contents via builder pattern (strawberry juice in tall glass)

Reference summary:
> A clear tall glass containing orange/yellow juice and several red
> strawberry pieces floating inside. THREE inventoried subjects: glass
> shell, liquid column, strawberries. The strawberries are visually
> dominant — they MUST be modeled, not collapsed into "decoration".

Pattern to reuse:

```javascript
function buildGlass(THREE) {
  const g = new THREE.Group();
  const mat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff, metalness: 0.0, roughness: 0.05,
    transmission: 0.95, ior: 1.5, transparent: true,
  });
  const rTop    = 0.18;
  const rBottom = 0.16;
  const h       = 0.62;
  // Open-top thin-walled cylinder shell:
  const shell = new THREE.Mesh(
    new THREE.CylinderGeometry(rTop, rBottom, h, 48, 1, true),
    mat,
  );
  shell.position.y = h / 2;
  g.add(shell);
  // Closed bottom disk so it doesn't read as a tube:
  const bottom = new THREE.Mesh(
    new THREE.CylinderGeometry(rBottom, rBottom, 0.01, 32),
    mat,
  );
  bottom.position.y = 0.005;
  g.add(bottom);
  // Expose internal cavity dims so caller can place liquid + pieces inside:
  g.userData.innerRTop    = rTop * 0.92;
  g.userData.innerRBottom = rBottom * 0.92;
  g.userData.innerH       = h - 0.02;
  g.userData.innerCenterY = (h - 0.02) / 2 + 0.01;
  return g;
}

function buildLiquid(THREE, innerRTop, innerRBottom, innerH, centerY, fillFrac = 0.78) {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({
    color: 0xf5a623,  // orange juice
    metalness: 0.0, roughness: 0.25, transparent: true, opacity: 0.92,
  });
  const liqH = innerH * fillFrac;
  const liq = new THREE.Mesh(
    new THREE.CylinderGeometry(innerRTop * 0.99, innerRBottom * 0.99, liqH, 48),
    mat,
  );
  // Sit liquid on the inner bottom, centered around glass interior:
  liq.position.y = centerY - innerH / 2 + liqH / 2;
  g.add(liq);
  g.userData.surfaceY = liq.position.y + liqH / 2;
  return g;
}

function buildStrawberry(THREE) {
  const g = new THREE.Group();
  const flesh = new THREE.MeshStandardMaterial({
    color: 0xd83a3a, metalness: 0.0, roughness: 0.45,
  });
  // Cone-like body (wider at top, pointed at the bottom):
  const body = new THREE.Mesh(
    new THREE.ConeGeometry(0.025, 0.05, 16),
    flesh,
  );
  body.rotation.z = Math.PI;  // point downward
  body.position.y = 0;
  g.add(body);
  // Small green leafy crown on top (single thin disk):
  const leaf = new THREE.Mesh(
    new THREE.ConeGeometry(0.022, 0.012, 6),
    new THREE.MeshStandardMaterial({ color: 0x4aa84a, roughness: 0.7 }),
  );
  leaf.position.y = 0.026;
  g.add(leaf);
  return g;
}

export default function generate(THREE) {
  const root = new THREE.Group();
  // inventory: glass, liquid, strawberry x5 (n = 7)
  // layout: liquid fills ~80% of glass interior; strawberries float inside
  //         the liquid at staggered heights

  const glass = buildGlass(THREE);
  const liquid = buildLiquid(
    THREE,
    glass.userData.innerRTop,
    glass.userData.innerRBottom,
    glass.userData.innerH,
    glass.userData.innerCenterY,
    0.78,
  );
  root.add(glass);
  root.add(liquid);

  // Five strawberries inside the liquid volume, deterministic placement:
  const N = 5;
  for (let i = 0; i < N; i++) {
    const s = buildStrawberry(THREE);
    const angle = (i / N) * Math.PI * 2;
    const radius = glass.userData.innerRTop * 0.55;
    s.position.set(
      Math.cos(angle) * radius,
      liquid.userData.surfaceY - 0.10 + (i % 2) * 0.06,  // staggered depth
      Math.sin(angle) * radius,
    );
    s.rotation.y = angle;
    root.add(s);
  }

  fitToUnitCube(THREE, root);
  return root;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
```

Pattern notes:
- Three builders, three subjects, one explicit inventory line.
- The glass exposes interior dimensions via `userData` so the liquid
  builder knows where to sit. The liquid exposes `surfaceY` so the
  strawberry placement knows where the surface is.
- Strawberries are placed via deterministic loop math (no `Math.random`)
  using `i / N * 2 * π` to fan them around — the same trick used for
  petals, spokes, repeated legs.
- The strawberries are SOLID floating meshes inside the liquid volume,
  NOT decoration dots on the glass surface. This is the most common
  failure mode for "X containing Y" — the contained Y gets collapsed
  into surface speckles.

### Example 4 — Ceramic floral decals on a curved vase surface

Reference summary:
> A glossy ceramic vase or pitcher with painted flowers and vines. The floral
> marks are flat glaze on the curved body, not separate 3D flowers floating in
> front of the object.

Pattern to reuse:

```javascript
// Place flat painted motifs on a rotational vase body.
const root = new THREE.Group(); // same group that contains the vase body
const decalMat = new THREE.MeshStandardMaterial({
  color: 0x6aa6d8,
  metalness: 0.0,
  roughness: 0.45,
  side: THREE.DoubleSide,
});
const stemMat = new THREE.MeshStandardMaterial({
  color: 0x4f7f5a,
  metalness: 0.0,
  roughness: 0.65,
});

function vaseRadiusAt(y) {
  // Match the lathe profile approximately; keep this in sync with the vessel.
  if (y < -0.25) return 0.16;
  if (y < 0.10) return 0.30;
  if (y < 0.35) return 0.24;
  return 0.18;
}

function surfacePose(angle, y, extra = 0.006) {
  const r = vaseRadiusAt(y) + extra;
  const normal = new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle)).normalize();
  const pos = new THREE.Vector3(normal.x * r, y, normal.z * r);
  const quat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 0, 1),
    normal
  );
  return { pos, quat, normal };
}

function addPetal(angle, y, localX, localY, sx, sy, rot, mat) {
  const { pos, quat } = surfacePose(angle, y);
  const petal = new THREE.Mesh(new THREE.CircleGeometry(0.035, 18), mat);
  petal.quaternion.copy(quat);
  petal.rotateZ(rot);
  petal.scale.set(sx, sy, 1);
  // Move in the decal's tangent plane after orientation.
  petal.position.copy(pos).add(
    new THREE.Vector3(localX, localY, 0).applyQuaternion(quat)
  );
  root.add(petal);
  return petal;
}

function addFlower(angle, y, size, mat) {
  for (let i = 0; i < 5; i++) {
    const a = i / 5 * Math.PI * 2;
    addPetal(
      angle, y,
      Math.cos(a) * size * 0.32,
      Math.sin(a) * size * 0.32,
      size * 1.00,
      size * 0.55,
      a,
      mat
    );
  }
  const { pos, quat } = surfacePose(angle, y, 0.008);
  const center = new THREE.Mesh(new THREE.CircleGeometry(size * 0.16, 14), mat);
  center.quaternion.copy(quat);
  center.position.copy(pos);
  root.add(center);
}

function addSurfaceVine(angle0, y0, angle1, y1) {
  const pts = [];
  for (let i = 0; i <= 8; i++) {
    const t = i / 8;
    const a = angle0 + (angle1 - angle0) * t;
    const y = y0 + (y1 - y0) * t + Math.sin(t * Math.PI) * 0.035;
    pts.push(surfacePose(a, y, 0.008).pos);
  }
  const vine = new THREE.Mesh(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 24, 0.004, 6, false),
    stemMat
  );
  root.add(vine);
}

addSurfaceVine(-0.75, -0.18, -0.20, 0.22);
addFlower(-0.70, -0.12, 0.80, decalMat);
addFlower(-0.42, 0.03, 0.55, decalMat);
addFlower(-0.22, 0.18, 0.38, decalMat);
```

Key idioms:
- Decorative texture is surface-bound: position = surface point + tiny normal
  offset, never a freestanding cluster in empty space.
- Flat petals use `CircleGeometry` / `ShapeGeometry` with tangent-plane scale,
  not bulky spheres unless the reference shows raised relief.
- `surfacePose(angle, y)` gives both position and orientation; every flower,
  leaf, and printed mark uses it.
- Vines/stems are tiny `TubeGeometry` curves whose points are all sampled on
  the same curved surface, not rods floating between motifs.
- Keep the vase body and decoration as one coherent group before calling
  `fitToUnitCube`.

These examples cover the most-failed patterns:
- Composing N-leg/N-spoke radial structures from a single geometry +
  position list.
- Lathe silhouettes with proper Vector2 control points.
- Multi-material, multi-part vehicles with helper functions, symmetric
  iteration, and correct wheel/tube geometry orientation.
- Seating furniture with distinct cushion modules, rolled arms, tufting,
  piping, slats, frames, and separate material regions.
- Surface-attached ceramic decals that do not float away from the body.
- Picking the right material class for the surface type.
- Mandatory normalization at end.

When the reference image shows something else, follow the same composition
discipline: single shared materials when uniform, explicit position lists
for symmetric arrays, Vector2 for any 2D-points API, When the reference image shows something else, follow the same composition
discipline: single shared materials when uniform, explicit position lists
for symmetric arrays, Vector2 for any 2D-points API, fitToUnitCube before
return.

## Winning idioms from prior leader-replacing miners

These five short modules each *beat* the previous round leader on a real prompt. Notice how each uses minimal primitives with explicit fitToUnitCube at the end. Borrow the material setups (metalness ≤ 0.6, transmission for glass, low roughness for polish) and the deterministic positioning patterns.

### Winning idiom — glass bottle (transmission + ior + thickness) (from cluster-mason)
```javascript
export default function generate(THREE) {
  const root = new THREE.Group();

  // Material: Blue glass with proper transmission for transparency
  const blue_glass_mat = new THREE.MeshPhysicalMaterial({
    color: 0x29b6f6,
    metalness: 0.0,
    roughness: 0.05,
    transmission: 0.95,
    ior: 1.5,
    transparent: true,
    thickness: 3.0,
    side: THREE.DoubleSide,
  });

  // Profile using SplineCurve for smooth curved shoulder transition
  // Neck is now ~25% of total height (was ~10%)
  const profileCurve = new THREE.SplineCurve([
    new THREE.Vector2(0.00, 0.00),  // Center bottom
    new THREE.Vector2(0.08, 0.02),  // Punt curve start
    new THREE.Vector2(0.28, 0.05),  // Inner bottom edge
    new THREE.Vector2(0.28, 0.55),  // Inner wall up (body height)
    new THREE.Vector2(0.24, 0.65),  // Inner shoulder curve start
    new THREE.Vector2(0.14, 0.72),  // Inner neck start
    new THREE.Vector2(0.14, 0.92),  // Inner neck top (taller neck ~25% of height)
    new THREE.Vector2(0.15, 0.92),  // Lip inner start
    new THREE.Vector2(0.18, 0.94),  // Lip inner curve
    new THREE.Vector2(0.22, 0.96),  // Lip top outer (thickened rounded rim)
    new THREE.Vector2(0.22, 0.94),  // Lip outer curve down
    new THREE.Vector2(0.16, 0.92),  // Neck outer top
    new THREE.Vector2(0.16, 0.72),  // Neck outer side (taller)
    new THREE.Vector2(0.24, 0.65),  // Outer shoulder curve (smooth transition)
    new THREE.Vector2(0.32, 0.55),  // Body outer side
    new THREE.Vector2(0.32, 0.05),  // Body outer bottom
    new THREE.Vector2(0.08, 0.02),  // Outer base curve
    new THREE.Vector2(0.00, 0.00),  // Close at center
  ]);

  const profile = profileCurve.getSpacedPoints(64);
  const bottle_geom = new THREE.LatheGeometry(profile, 48);
  const bottle = new THREE.Mesh(bottle_geom, blue_glass_mat);

  root.add(bottle);

  fitToUnitCube(THREE, root);
  return root;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
```

### Winning idiom — wheeled / metallic object (from cluster-mason)
```javascript
export default function generate(THREE) {
  const root = new THREE.Group();

  // Materials
  const metalMat = new THREE.MeshStandardMaterial({
    color: 0x2a4a9c,
    metalness: 0.6,
    roughness: 0.2,
  });

  const gemMat = new THREE.MeshPhysicalMaterial({
    color: 0x4488ff,
    metalness: 0.0,
    roughness: 0.05,
    transmission: 0.95,
    ior: 1.5,
    transparent: true,
  });

  // 1. Band - slender ring shank (~10-15% of height), not thick donut
  const bandGeom = new THREE.TorusGeometry(0.30, 0.05, 24, 64);
  const band = new THREE.Mesh(bandGeom, metalMat);
  band.rotation.x = Math.PI / 2;
  root.add(band);

  // 2. Bezel/Setting - substantial rounded setting that integrates with band shoulders
  // Use lathe for smooth signet profile that rises from band
  const bezelProfile = [
    new THREE.Vector2(0.00, 0.00),
    new THREE.Vector2(0.18, 0.00),
    new THREE.Vector2(0.20, 0.04),
    new THREE.Vector2(0.19, 0.08),
    new THREE.Vector2(0.17, 0.10),
    new THREE.Vector2(0.00, 0.10),
  ];
  const bezelGeom = new THREE.LatheGeometry(bezelProfile, 32);
  const bezel = new THREE.Mesh(bezelGeom, metalMat);
  bezel.rotation.x = Math.PI / 2;
  bezel.position.set(0, 0.05, -0.30);
  root.add(bezel);

  // 3. Gemstone - large oval faceted gem, set INTO bezel (30% enclosed)
  const stoneGeom = new THREE.IcosahedronGeometry(0.14, 2);
  const stone = new THREE.Mesh(stoneGeom, gemMat);
  stone.scale.set(1.3, 0.8, 1.1);
  // Position so lower 30% is enclosed by bezel
  stone.position.set(0, 0.12, -0.30);
  stone.rotation.y = Math.PI / 6;
  stone.rotation.z = Math.PI / 12;
  stone.rotation.x = Math.PI / 8;
  root.add(stone);

  fitToUnitCube(THREE, root);
  return root;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
```

### Winning idiom — matte ceramic / wood (from solid-mango)
```javascript
export default function generate(THREE) {
  const root = new THREE.Group();

  // Materials
  const baseMat = new THREE.MeshStandardMaterial({
    color: 0xe8dcc8,
    metalness: 0.0,
    roughness: 0.7,
  });

  const shadeMat = new THREE.MeshStandardMaterial({
    color: 0xfff9f0,
    metalness: 0.0,
    roughness: 0.4,
    emissive: 0xffeebb,
    emissiveIntensity: 1.2,
  });

  const cordMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    metalness: 0.0,
    roughness: 0.6,
  });

  // Base: Flat circular disc
  const baseGeom = new THREE.CylinderGeometry(0.28, 0.28, 0.04, 32);
  const base = new THREE.Mesh(baseGeom, baseMat);
  base.position.y = 0.02;
  root.add(base);

  // Joint: Small connector cylinder
  const jointGeom = new THREE.CylinderGeometry(0.06, 0.06, 0.03, 16);
  const joint = new THREE.Mesh(jointGeom, baseMat);
  joint.position.y = 0.055;
  root.add(joint);

  // Shade: Tall cylinder, tilted
  const shadeHeight = 0.65;
  const shadeRadius = 0.14;
  const shadeGeom = new THREE.CylinderGeometry(shadeRadius, shadeRadius, shadeHeight, 32);
  const shade = new THREE.Mesh(shadeGeom, shadeMat);
  
  // Position shade on top of joint, then tilt
  // The pivot for tilt should be near the bottom of the shade
  const shadePivot = new THREE.Group();
  shadePivot.position.set(0, 0.07, 0);
  
  shade.position.y = shadeHeight / 2;
  shadePivot.add(shade);
  
  // Tilt backwards slightly (around X axis)
  shadePivot.rotation.x = -Math.PI / 9; // ~20 degrees
  
  root.add(shadePivot);

  // Cord: Thin tube exiting from the back of the base
  const cordPath = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0, 0.02, -0.28), // Start at back edge of base
    new THREE.Vector3(0, 0.01, -0.45), // Drop slightly
    new THREE.Vector3(0.3, 0.0, -0.60), // Curve to the right and back
    new THREE.Vector3(0.6, 0.0, -0.70), // Continue away
  ]);
  
  const cordGeom = new THREE.TubeGeometry(cordPath, 20, 0.006, 8, false);
  const cord = new THREE.Mesh(cordGeom, cordMat);
  root.add(cord);

  // Normalization helper
  fitToUnitCube(THREE, root);
  return root;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
```

### Winning idiom — polished marble / ceramic vase (from cautious-adventure)
```javascript
export default function generate(THREE) {
  const root = new THREE.Group();

  // Material: Polished white marble / ceramic
  // Using standard material with low metalness and moderate roughness.
  // Color is off-white to simulate stone.
  const marbleMat = new THREE.MeshStandardMaterial({
    color: 0xe8e8e8,
    metalness: 0.0,
    roughness: 0.35,
  });

  // --- Base ---
  // Two-tiered cylindrical base.
  // Bottom tier: slightly narrower, thicker.
  const baseBottomGeom = new THREE.CylinderGeometry(0.24, 0.24, 0.06, 32);
  const baseBottom = new THREE.Mesh(baseBottomGeom, marbleMat);
  baseBottom.position.y = 0.03; // Half height
  root.add(baseBottom);

  // Top tier: wider, thinner disc.
  const baseTopGeom = new THREE.CylinderGeometry(0.30, 0.30, 0.05, 32);
  const baseTop = new THREE.Mesh(baseTopGeom, marbleMat);
  baseTop.position.y = 0.06 + 0.025; // Bottom height + half top height
  root.add(baseTop);

  // --- Sculpture ---
  // Twisted toroidal ring.
  // Using TorusKnotGeometry with p=1, q=2 to create a double-twist loop.
  // Radius: 0.24 (fits on base)
  // Tube: 0.09 (thick, substantial volume)
  // Segments: High count for smoothness.
  const sculptureRadius = 0.24;
  const sculptureTube = 0.09;
  const sculptureGeom = new THREE.TorusKnotGeometry(
    sculptureRadius,
    sculptureTube,
    128, // tubularSegments
    32,  // radialSegments
    1,   // p (winds around major axis)
    2    // q (winds around minor axis - creates the twist)
  );

  const sculpture = new THREE.Mesh(sculptureGeom, marbleMat);
  
  // Position: Centered above the base.
  // Base total height is 0.11.
  // Sculpture bottom should touch base top.
  // Sculpture center Y = Base Top Y + Sculpture Radius.
  sculpture.position.y = 0.11 + sculptureRadius;
  
  // Orientation: TorusKnot lies in XY plane by default.
  // We want it standing vertically (like a wheel), so rotate around X axis by 90 degrees.
  sculpture.rotation.x = Math.PI / 2;
  
  root.add(sculpture);

  fitToUnitCube(THREE, root);
  return root;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
```

### Winning idiom — wax / candle composition (from cautious-adventure)
```javascript
export default function generate(THREE) {
  const root = new THREE.Group();

  // Materials
  const waxMat = new THREE.MeshStandardMaterial({
    color: 0x66b2e6,
    metalness: 0.0,
    roughness: 0.55,
  });

  const wickMat = new THREE.MeshStandardMaterial({
    color: 0x2a4a7a,
    metalness: 0.0,
    roughness: 0.9,
  });

  const flameMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    metalness: 0.0,
    roughness: 0.1,
    transmission: 0.85,
    ior: 1.33,
    transparent: true,
    opacity: 0.9,
    emissive: 0xffdd88,
    emissiveIntensity: 1.5,
  });

  // Candle Body - light sky blue with subtle wax texture
  const candleHeight = 0.8;
  const candleRadius = 0.15;
  const candleGeom = new THREE.CylinderGeometry(candleRadius, candleRadius, candleHeight, 32);
  const candle = new THREE.Mesh(candleGeom, waxMat);
  root.add(candle);

  // Wick - dark blue to match candle body tone
  const wickHeight = 0.04;
  const wickRadius = 0.01;
  const wickGeom = new THREE.CylinderGeometry(wickRadius, wickRadius, wickHeight, 8);
  const wick = new THREE.Mesh(wickGeom, wickMat);
  wick.position.y = candleHeight / 2 + wickHeight / 2;
  root.add(wick);

  // Flame - glowing translucent teardrop with smooth gradient
  // Profile: bottom center -> base edge -> widest point -> tip (smooth curve)
  const flameProfile = [
    new THREE.Vector2(0.0, 0.0),
    new THREE.Vector2(0.045, 0.0),
    new THREE.Vector2(0.065, 0.12),
    new THREE.Vector2(0.035, 0.22),
    new THREE.Vector2(0.0, 0.28),
  ];
  const flameGeom = new THREE.LatheGeometry(flameProfile, 32);
  const flame = new THREE.Mesh(flameGeom, flameMat);
  flame.position.y = candleHeight / 2 + wickHeight;
  root.add(flame);

  fitToUnitCube(THREE, root);
  return root;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
```
"""
