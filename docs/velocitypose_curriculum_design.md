# VelocityPose Curriculum Design - Physical Constraints Analysis

## 🤖 Unitree Go2 Physical Parameters

### Joint Configuration
```
Default Standing Pose (from config):
- Hip joints:   0.0 rad (both sides)
- Thigh joints: 0.8 rad (47.9°)
- Calf joints: -1.5 rad (-85.9°)
- Initial height: 0.38m → Settles to ~0.33m
```

### Leg Dimensions (Estimated from URDF)
```
Approximate Leg Segments:
- Hip offset (lateral):  ~0.08m (8cm)
- Upper leg (thigh):     ~0.20m (20cm)
- Lower leg (calf):      ~0.20m (20cm)
- Total leg length:      ~0.40m (40cm)
```

### Joint Limits (Typical for Go2)
```
Hip (abduction/adduction):   ±0.8 rad  (±45.8°)
Thigh (flexion/extension):   [-1.0, 2.5] rad
Calf (flexion/extension):    [-2.7, -0.9] rad
```

---

## 📐 Kinematic Constraints Analysis

### 1. Height Range Analysis

#### Minimum Height (~0.18m)
**Configuration**: Legs fully bent
- Thigh: ~1.2 rad (68°)
- Calf: -2.5 rad (-143°)
- **Physical limit**: Body touches ground, feet may slip
- **Safe minimum**: 0.20m

#### Default Height (0.33m)
**Configuration**: Standard standing
- Thigh: 0.8 rad (45.8°)
- Calf: -1.5 rad (-85.9°)
- **Stability**: Excellent
- **Reserve**: Good margin for movement

#### Maximum Height (~0.42m)
**Configuration**: Legs nearly straight
- Thigh: 0.3 rad (17°)
- Calf: -1.0 rad (-57°)
- **Physical limit**: Leg fully extended, minimal control authority
- **Safe maximum**: 0.40m

**Conclusion**: 
✅ **Safe height range**: [0.22m, 0.40m]  
✅ **Optimal training range**: [0.28m, 0.38m] (±5cm from default)

---

### 2. Orientation (Roll/Pitch) Constraints

#### Roll Analysis (Side-to-side tilt)

The robot can compensate roll by:
- Extending legs on one side
- Compressing legs on the other side

**Physical Calculation**:
```
Hip spacing: ~0.31m (front-to-front or rear-to-rear)
Max height difference: 0.40m - 0.22m = 0.18m
Max roll angle: arctan(0.18 / 0.31) ≈ 30° (0.52 rad)
```

**BUT with coupling constraints**:
- At default height (0.33m):
  - One side at 0.40m, other at 0.26m
  - Roll: arctan(0.14 / 0.31) ≈ 24° (0.42 rad)
  
- At maximum height (0.40m):
  - Cannot roll much (already at max extension)
  - Safe roll: ±10° (±0.17 rad)

- At minimum height (0.22m):
  - Cannot roll much (already at min extension)
  - Safe roll: ±10° (±0.17 rad)

**Conclusion**: 
✅ **Safe roll range at h=0.33m**: ±15° (±0.26 rad)  
✅ **Conservative training**: ±10° (±0.17 rad)

---

#### Pitch Analysis (Front-to-back tilt)

Similar to roll, but affects gait more severely:
- Forward pitch: Shifts weight to front legs
- Backward pitch: Shifts weight to rear legs
- Affects stability during locomotion

**Physical Calculation**:
```
Body length: ~0.60m (front hip to rear hip)
Max height difference: 0.18m
Max pitch angle: arctan(0.18 / 0.60) ≈ 17° (0.30 rad)
```

**Dynamic constraints**:
- During walking: Weight distribution critical
- Pitch changes affect gait phase
- Large pitch reduces stability margin

**Conclusion**: 
✅ **Safe pitch range**: ±12° (±0.21 rad)  
✅ **Conservative training**: ±8° (±0.14 rad)

---

### 3. Coupled Constraints (Height + Orientation)

**Critical insight**: Height and orientation are NOT independent!

#### Constraint Matrix

| Height (m) | Safe Roll (°) | Safe Pitch (°) | Rationale |
|------------|---------------|----------------|-----------|
| 0.22 (min) | ±5°  | ±5°  | Already at limit, minimal adjustment room |
| 0.28       | ±12° | ±10° | Good range, stable |
| 0.33 (def) | ±15° | ±12° | Best stability, max control authority |
| 0.38       | ±12° | ±10° | Reduced range due to extension |
| 0.40 (max) | ±8°  | ±6°  | Limited range, near singularity |

#### Coupling Formula (Conservative)

For safety, use elliptical constraint:
```
(height - 0.33)² / 0.07²  +  roll² / 0.26²  +  pitch² / 0.21²  ≤ 1
```

This ensures we don't command extreme combinations like:
- ❌ h=0.40m + roll=15° (impossible)
- ❌ h=0.22m + pitch=12° (unstable)
- ✅ h=0.33m + roll=10° + pitch=8° (safe)

---

## 🎓 Recommended Curriculum Stages

### Stage 0: Baseline (0-2000 iterations)
**Goal**: Learn velocity tracking only

```python
Height:  0.33m (fixed, no variation)
Roll:    0.0° (fixed)
Pitch:   0.0° (fixed)

# Rewards active:
- track_lin_vel_xy_exp: 3.0
- track_ang_vel_z_exp: 1.5
- track_height_exp: 2.0 (always rewarding 0.33m)
- track_orientation_exp: 1.0 (always rewarding 0°)
```

**Expected outcome**: Stable walking on rough terrain

---

### Stage 1: Introduce Small Height Variation (2000-5000 iterations)
**Goal**: Learn to adjust height while walking

```python
Height:  [0.30, 0.36]  # ±3cm (9% variation)
Roll:    0.0° (fixed)
Pitch:   0.0° (fixed)

# Physics check:
# Max leg length change: 3cm
# Achievable with minimal joint adjustment
# Safe for all terrains
```

**Validation**:
- Height tracking error < 2cm
- Velocity tracking maintained
- No increase in falls

---

### Stage 2: Add Modest Roll (5000-8000 iterations)
**Goal**: Learn lateral weight shifting

```python
Height:  [0.30, 0.36]  # Keep previous range
Roll:    [-8°, +8°] = [-0.14, +0.14] rad
Pitch:   0.0° (fixed)

# Physics check at h=0.33m, roll=8°:
# Left side height: 0.33 + 0.31*sin(8°) = 0.374m ✅
# Right side height: 0.33 - 0.31*sin(8°) = 0.286m ✅
# Both within [0.22, 0.40] safe range
```

**Validation**:
- Roll tracking error < 3°
- Height tracking maintained
- Lateral stability preserved

---

### Stage 3: Add Modest Pitch (8000-12000 iterations)
**Goal**: Learn forward/backward weight management

```python
Height:  [0.30, 0.36]
Roll:    [-8°, +8°]
Pitch:   [-6°, +6°] = [-0.10, +0.10] rad

# Physics check at h=0.33m, pitch=6°:
# Front legs: 0.33 + 0.30*sin(6°) = 0.361m ✅
# Rear legs: 0.33 - 0.30*sin(6°) = 0.299m ✅
```

**Validation**:
- Pitch tracking error < 3°
- Gait remains stable
- No frequent tripping

---

### Stage 4: Expand Height Range (12000-16000 iterations)
**Goal**: Full height control authority

```python
Height:  [0.28, 0.38]  # ±5cm (15% variation)
Roll:    [-8°, +8°]
Pitch:   [-6°, +6°]

# Physics check:
# All combinations within safe envelope
# Max coupled error from coupling formula: 0.85 < 1.0 ✅
```

---

### Stage 5: Full Control (16000+ iterations)
**Goal**: Maximum operational envelope

```python
Height:  [0.26, 0.40]  # ±7cm, but...
Roll:    [-10°, +10°] = [-0.17, +0.17] rad
Pitch:   [-8°, +8°] = [-0.14, +0.14] rad

# WITH coupling constraint:
# Reject commands where:
# (h-0.33)²/0.07² + roll²/0.26² + pitch²/0.21² > 1.0
```

**Implementation**:
Add command rejection/clamping in command generator:
```python
def _reject_unsafe_commands(self, height, roll, pitch):
    """Reject physically infeasible command combinations"""
    h_norm = (height - 0.33) / 0.07
    r_norm = roll / 0.26
    p_norm = pitch / 0.21
    
    constraint = h_norm**2 + r_norm**2 + p_norm**2
    
    if constraint > 1.0:
        # Scale back to boundary
        scale = torch.sqrt(constraint)
        height = 0.33 + (height - 0.33) / scale
        roll = roll / scale
        pitch = pitch / scale
    
    return height, roll, pitch
```

---

## 📊 Summary Table

| Stage | Iterations | Height (m) | Roll (°) | Pitch (°) | Difficulty |
|-------|-----------|------------|----------|-----------|------------|
| 0 | 0-2k      | 0.33 (fixed) | 0 (fixed) | 0 (fixed) | ⭐ Easy |
| 1 | 2k-5k     | [0.30, 0.36] | 0 (fixed) | 0 (fixed) | ⭐⭐ Low |
| 2 | 5k-8k     | [0.30, 0.36] | ±8° | 0 (fixed) | ⭐⭐⭐ Medium |
| 3 | 8k-12k    | [0.30, 0.36] | ±8° | ±6° | ⭐⭐⭐⭐ High |
| 4 | 12k-16k   | [0.28, 0.38] | ±8° | ±6° | ⭐⭐⭐⭐⭐ Very High |
| 5 | 16k+      | [0.26, 0.40] | ±10° | ±8° | 🔥 Maximum |

---

## ⚠️ Safety Checks During Training

### Automatic Command Validation

Add to command generator:
```python
def _validate_command(self, height, roll, pitch):
    """Validate command is physically achievable"""
    
    # Check 1: Individual limits
    height = torch.clamp(height, 0.22, 0.40)
    roll = torch.clamp(roll, -0.17, 0.17)
    pitch = torch.clamp(pitch, -0.14, 0.14)
    
    # Check 2: Coupled constraint
    height, roll, pitch = self._reject_unsafe_commands(height, roll, pitch)
    
    # Check 3: Height-dependent orientation limits
    if height > 0.37:  # Near max extension
        roll = torch.clamp(roll, -0.14, 0.14)  # Reduce to ±8°
        pitch = torch.clamp(pitch, -0.10, 0.10)  # Reduce to ±6°
    
    if height < 0.25:  # Near min compression
        roll = torch.clamp(roll, -0.09, 0.09)  # Reduce to ±5°
        pitch = torch.clamp(pitch, -0.09, 0.09)  # Reduce to ±5°
    
    return height, roll, pitch
```

### Training Metrics to Monitor

```python
# Add to curriculum evaluation:
metrics = {
    "height_tracking_error": torch.abs(actual_height - cmd_height).mean(),
    "roll_tracking_error": torch.abs(actual_roll - cmd_roll).mean(),
    "pitch_tracking_error": torch.abs(actual_pitch - cmd_pitch).mean(),
    "fall_rate": (terminations / total_steps).mean(),
    "joint_limit_violations": (joint_pos > limits).sum(),
    "command_rejection_rate": rejected_commands / total_commands,
}

# Progression criteria:
if (metrics["height_tracking_error"] < 0.02 and  # < 2cm
    metrics["roll_tracking_error"] < 0.05 and     # < 3°
    metrics["pitch_tracking_error"] < 0.05 and    # < 3°
    metrics["fall_rate"] < 0.05):                 # < 5%
    # Ready for next stage
    advance_curriculum()
```

---

## 🎯 Implementation Priority

1. **✅ Immediate**: Add coupling constraint to command generator
2. **✅ High**: Implement Stage 0-3 curriculum (covers 80% use cases)
3. **⚠️ Medium**: Add automatic validation and safety checks
4. **💡 Future**: Implement Stage 4-5 for advanced scenarios

This ensures:
- No physically impossible commands
- Smooth learning progression
- Safe operation throughout training
- Maximum final capability within physical limits
