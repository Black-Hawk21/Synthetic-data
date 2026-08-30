# Demo image-generation prompts

Manual prompts for generating 2 damage-claim images per object from a real
reference photo, matching the `img2img_conditioned_two_angle` pattern already
used by the Red Team (see `aegis/attacks/taxonomy.py` and
`aegis/redteam/prompts.py`): image 2 is generated with `reference_b64` set to
image 1 (or directly to the real photo) so damage position, color, and
lighting stay consistent across angles.

## 1. Sony ULT WEAR headphones

Reference: a real photo of the undamaged headphones.

**Image 1 -- front damage close-up:**

> Photo-realistic phone-camera photo of these exact Sony ULT WEAR wireless
> headphones, same colorway and logo placement as the reference photo. Left
> ear cup has a visible crack through the padding with foam exposed,
> consistent with being dropped. Shot straight-on, slightly off-center
> framing, single-window indoor lighting, natural shadow, subtle sensor
> noise, no studio lighting.

**Image 2 -- 45-degree side angle (condition on image 1):**

> Same Sony ULT WEAR headphones as the reference photo, 45-degree side angle
> showing the headband and the cracked left ear cup from a different
> perspective. Crack shape and position must match the reference exactly,
> same lighting direction and color temperature, natural handheld
> phone-photo look.

## 2. Pizza delivery box

Picked deliberately, per the same "what makes an object work well" reasoning as the
headphones: a rigid, fixed geometry (rectangular box, flaps, corners) gives the
object-identity check concrete landmarks, a printed logo gives it something to
verify placement/font on, and a crushed corner is a simple, iconic damage shape
that's easy to keep visually consistent across two generated angles.

Reference: a real photo of an undamaged box if you have one on hand. If not,
generate a plausible reference first (keep the branding generic/fictional -- avoid
naming a real chain):

> Photo-realistic phone-camera photo of a closed pizza delivery box, plain
> cardboard with a simple printed logo on the lid, sitting on a kitchen counter,
> natural indoor lighting, undamaged, straight-on angle.

Then use that output as the reference for both images below, same workflow as the
headphones.

**Image 1 -- front damage close-up:**

> Photo-realistic phone-camera photo of this exact pizza delivery box, same
> cardboard color, branding, and logo placement as the reference photo, now
> visibly damaged: one corner of the box crushed and caved in, the lid slightly
> popped open from the crush, cheese and toppings visibly slid toward the
> crushed corner. Shot straight-on on a kitchen counter, overhead indoor
> lighting, realistic phone-camera grain, no studio lighting.

**Image 2 -- top-down angle (condition on image 1):**

> Same pizza delivery box as the reference photo, top-down angle with the lid
> open, showing the crushed corner and the pizza inside with toppings shifted
> toward that same corner. Crush shape, position, and severity must match image
> 1 exactly -- same corner, same extent. Same lighting direction and color
> temperature, natural handheld phone-photo look.

Note on "crush shape/position must match exactly": we've since confirmed (Sony
example) that the detection pipeline cannot reliably verify this itself even when
it's wrong -- two top-tier models both missed an obvious shape mismatch a human
caught instantly. Keeping the generated damage genuinely consistent here means
you get the same honest "hard case" demo as Sony, rather than an accidentally easy
naive-fraud giveaway. If you want an easy-to-catch contrast case instead, generate
the two angles independently (no `reference_b64` on image 2) and skip the "must
match exactly" instruction -- that reliably produces visible drift.

## Usage

```python
img1 = image_provider.generate_image(prompt1, reference_b64=real_photo_b64)
img2 = image_provider.generate_image(prompt2, reference_b64=img1)  # or reference_b64=real_photo_b64
```
