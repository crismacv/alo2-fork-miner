"""R10 few-shot examples — winner-style monolithic code.

Two examples taken from the R8 winner-candidate (5HgGDgMf...) submissions:
a pocket watch (LatheGeometry case + dial + markers loop + bezel + bow +
glass cover) and a calculator (ExtrudeGeometry wedge body + screen +
solar cell + keypad grid).

Style notes the model should pick up from these:
- Materials block first, dimensions block next, body, then features.
- Each region of the reference becomes its own Mesh added to root.
- Repeated parts use a for-loop with deterministic `i / N * 2 * Math.PI`.
- Non-circular silhouettes use ExtrudeGeometry with a custom Shape.
- LatheGeometry for rotationally symmetric vessels, rotated to face +Z.
- No builder functions, no inventory comments — just clean inline code.
"""

FEW_SHOT_EXAMPLES = '''\
## Worked examples — study the style, then apply to your reference

### Example 1 — Pocket watch (Lathe case, torus bezel, dial markers loop, attached bow ring)

Reference summary:
> Round gold pocket watch, white dial with Roman-numeral markers (12 / 3 /
> 6 / 9 are heavier), hour and minute hand pivoting at center, glass dome
> over the face, gold bow ring at the top attached to the case.

```javascript
export default function generate(THREE) {
  const root = new THREE.Group();

  // --- Materials ---
  const goldMat = new THREE.MeshStandardMaterial({
    color: 0xd4af37, metalness: 0.6, roughness: 0.25,
  });
  const dialMat = new THREE.MeshStandardMaterial({
    color: 0xf5f5f0, metalness: 0.0, roughness: 0.4,
  });
  const glassMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff, metalness: 0.0, roughness: 0.05,
    transmission: 0.95, ior: 1.5, transparent: true,
  });
  const blackMat = new THREE.MeshStandardMaterial({
    color: 0x1a1a1a, metalness: 0.5, roughness: 0.4,
  });

  // --- Dimensions ---
  const caseRadius = 0.45;
  const caseDepth  = 0.12;
  const dialRadius = 0.41;
  const bezelWidth = 0.035;

  // --- 1. Case body (Lathe for rounded profile, then rotated to face +Z) ---
  const caseProfile = [
    new THREE.Vector2(0, -caseDepth / 2),
    new THREE.Vector2(caseRadius * 0.95, -caseDepth / 2),
    new THREE.Vector2(caseRadius, 0),
    new THREE.Vector2(caseRadius * 0.95, caseDepth / 2),
    new THREE.Vector2(0, caseDepth / 2),
  ];
  const caseMesh = new THREE.Mesh(new THREE.LatheGeometry(caseProfile, 32), goldMat);
  caseMesh.rotation.x = Math.PI / 2;
  root.add(caseMesh);

  // --- 2. Bezel ring (front rim) ---
  const bezel = new THREE.Mesh(
    new THREE.TorusGeometry(caseRadius, bezelWidth, 16, 32), goldMat,
  );
  bezel.position.z = caseDepth / 2 - 0.01;
  root.add(bezel);

  // --- 3. Dial (white face) ---
  const dial = new THREE.Mesh(
    new THREE.CylinderGeometry(dialRadius, dialRadius, 0.005, 32), dialMat,
  );
  dial.rotation.x = Math.PI / 2;
  dial.position.z = caseDepth / 2 + 0.005;
  root.add(dial);

  // --- 4. Markers (12 bars, cardinal positions thicker) ---
  const markerGroup = new THREE.Group();
  markerGroup.position.z = caseDepth / 2 + 0.008;
  for (let i = 0; i < 12; i++) {
    const angle = (i / 12) * Math.PI * 2;
    const isCardinal = (i % 3 === 0);
    const thickness = isCardinal ? 0.015 : 0.008;
    const length    = isCardinal ? 0.04  : 0.025;
    const rOuter = dialRadius * 0.95;
    const midR = rOuter - length / 2;
    const bar = new THREE.Mesh(
      new THREE.BoxGeometry(thickness, length, 0.002), blackMat,
    );
    bar.position.set(Math.cos(angle) * midR, Math.sin(angle) * midR, 0);
    bar.rotation.z = angle - Math.PI / 2;
    markerGroup.add(bar);
  }
  root.add(markerGroup);

  // --- 5. Hour + minute hands ---
  const hourHand = new THREE.Mesh(
    new THREE.BoxGeometry(0.015, 0.18, 0.004), blackMat,
  );
  hourHand.position.set(0, 0.09, caseDepth / 2 + 0.012);
  root.add(hourHand);
  const minuteHand = new THREE.Mesh(
    new THREE.BoxGeometry(0.010, 0.28, 0.004), blackMat,
  );
  minuteHand.position.set(0.05, 0.10, caseDepth / 2 + 0.014);
  minuteHand.rotation.z = -0.4;
  root.add(minuteHand);
  const pivot = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 16), goldMat);
  pivot.position.z = caseDepth / 2 + 0.018;
  root.add(pivot);

  // --- 6. Bow ring (attached to top of case, overlap by ~30%) ---
  const bowGroup = new THREE.Group();
  bowGroup.position.set(0, caseRadius * 0.95, 0);
  const stem = new THREE.Mesh(
    new THREE.CylinderGeometry(0.025, 0.025, 0.08, 16), goldMat,
  );
  stem.position.y = 0.04;
  bowGroup.add(stem);
  const bowRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.10, 0.015, 16, 32), goldMat,
  );
  bowRing.position.y = 0.10;
  bowGroup.add(bowRing);
  root.add(bowGroup);

  // --- 7. Glass cover ---
  const glass = new THREE.Mesh(
    new THREE.CylinderGeometry(dialRadius + 0.005, dialRadius + 0.005, 0.005, 32),
    glassMat,
  );
  glass.rotation.x = Math.PI / 2;
  glass.position.z = caseDepth / 2 + 0.015;
  root.add(glass);

  fitToUnitCube(THREE, root);
  return root;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3(); box.getSize(size);
  const center = new THREE.Vector3(); box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
```

Style notes:
- Materials defined once at the top, reused for body, bezel, markers etc.
- Lathe profile is just five Vector2 points; rotation.x = π/2 turns the
  axis-symmetric body to face the camera (+Z).
- Markers loop uses `i / 12 * 2π` for deterministic radial placement.
- The bow ring overlaps the top of the case (no float).
- Glass uses MeshPhysicalMaterial with transmission so the dial shows through.


### Example 2 — Blue calculator (ExtrudeGeometry wedge body, screen, solar cell, 4×4 keypad)

Reference summary:
> Blue wedge-shaped pocket calculator. Larger thickness at the back, thinner
> at the front (typewriter-style angle). LCD screen window at top half, small
> solar cell to the right of the screen, 4×4 grid of green keys filling the
> bottom half. Black plastic frame around the screen.

```javascript
export default function generate(THREE) {
  const root = new THREE.Group();

  // --- Materials ---
  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0x0055ff, roughness: 0.6, metalness: 0.1,
  });
  const keyMat = new THREE.MeshStandardMaterial({
    color: 0x44cc44, roughness: 0.6, metalness: 0.0,
  });
  const screenMat = new THREE.MeshStandardMaterial({
    color: 0x334433, roughness: 0.2, metalness: 0.0,
  });
  const screenFrameMat = new THREE.MeshStandardMaterial({
    color: 0x111111, roughness: 0.5, metalness: 0.0,
  });
  const solarMat = new THREE.MeshStandardMaterial({
    color: 0xaa2222, roughness: 0.3, metalness: 0.1,
  });

  // --- Dimensions ---
  const length    = 1.6;
  const width     = 0.9;
  const thickBack = 0.35;
  const thickFront= 0.10;

  // --- 1. Wedge body via ExtrudeGeometry. ---
  // Side profile: X = depth (front→back), Y = height.
  // Trapezoid wedge: back edge tall, front edge short.
  const shape = new THREE.Shape();
  shape.moveTo(-length / 2, 0);
  shape.lineTo( length / 2, 0);
  shape.lineTo( length / 2, thickFront);
  shape.lineTo(-length / 2, thickBack);
  shape.lineTo(-length / 2, 0);
  const body = new THREE.Mesh(
    new THREE.ExtrudeGeometry(shape, {
      depth: width, bevelEnabled: true, bevelThickness: 0.02,
      bevelSize: 0.02, bevelSegments: 4, curveSegments: 4,
    }),
    bodyMat,
  );
  body.rotation.x = -Math.PI / 2;            // Lay flat with Y up.
  body.position.set(0, 0, 0);
  root.add(body);

  // The face plane that the screen / keys sit on is the SLOPED top.
  // Top face spans from (x=-length/2, y=thickBack) to (x=+length/2, y=thickFront).
  // Compute helpers for placement on that slope:
  const slope = (thickFront - thickBack) / length;
  function onFace(xLocal, zLocal, lift) {
    // xLocal in [-length/2, length/2], zLocal in [-width/2, width/2], lift = small +offset above face.
    return new THREE.Vector3(
      xLocal,
      thickBack + slope * (xLocal + length / 2) + lift,
      zLocal,
    );
  }

  // --- 2. Screen frame + LCD (top portion of the face) ---
  const screenW = length * 0.65;
  const screenH = width * 0.32;
  const screenCenterX = -length * 0.05; // shifted left of center
  const screenCenterZ = -width * 0.28;  // upper band of face
  const frame = new THREE.Mesh(
    new THREE.BoxGeometry(screenW + 0.04, 0.02, screenH + 0.04),
    screenFrameMat,
  );
  frame.position.copy(onFace(screenCenterX, screenCenterZ, 0.005));
  root.add(frame);
  const lcd = new THREE.Mesh(
    new THREE.BoxGeometry(screenW, 0.015, screenH),
    screenMat,
  );
  lcd.position.copy(onFace(screenCenterX, screenCenterZ, 0.012));
  root.add(lcd);

  // --- 3. Solar cell (small dark red panel, right of the screen) ---
  const solar = new THREE.Mesh(
    new THREE.BoxGeometry(length * 0.18, 0.015, width * 0.12),
    solarMat,
  );
  solar.position.copy(onFace(length * 0.30, screenCenterZ, 0.012));
  root.add(solar);

  // --- 4. Keypad (4×4 grid of green keys, lower band of face) ---
  const cols = 4, rows = 4;
  const keyW = length * 0.16;
  const keyD = width * 0.16;
  const keyH = 0.04;
  const padX0 = -length * 0.30;
  const padZ0 =  width * 0.05;
  const stepX = length * 0.18;
  const stepZ = width * 0.20;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const xL = padX0 + c * stepX;
      const zL = padZ0 + r * stepZ;
      const key = new THREE.Mesh(
        new THREE.BoxGeometry(keyW, keyH, keyD), keyMat,
      );
      const pos = onFace(xL, zL, keyH / 2);
      key.position.copy(pos);
      root.add(key);
    }
  }

  fitToUnitCube(THREE, root);
  return root;
}

function fitToUnitCube(THREE, root) {
  const box = new THREE.Box3().setFromObject(root);
  const size = new THREE.Vector3(); box.getSize(size);
  const center = new THREE.Vector3(); box.getCenter(center);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const scale = 0.95 / maxDim;
  root.scale.setScalar(scale);
  root.position.set(-center.x * scale, -center.y * scale, -center.z * scale);
}
```

Style notes:
- ExtrudeGeometry with a 5-point Shape gives the trapezoidal wedge body
  with one motion — no need to stack two boxes at an angle.
- `bevelEnabled: true` softens all edges of the body (matches a real
  consumer-electronics finish without writing 12 separate edge meshes).
- A small `onFace(xLocal, zLocal, lift)` helper computes the slope-aware
  position so screen / solar / keys all sit flush on the sloped face.
- 4×4 keypad uses nested for-loop with named step constants — easy to
  tweak.
- Materials capped at metalness 0.5 (solar is the only metallic-ish part).
'''
