# Sweet Spot Simulation

A browser-based listening lab for spatial audio in cinemas and car cabins. Pick a room, a speaker layout and a seat; every speaker's sound reaches your seat with its real travel time, level and angle, then is rendered binaurally (HRTF) for **headphones**.

## Run it

Open `index.html` in a browser (Chrome, Edge, Firefox or Safari) and press **Start sound**. No build step and no dependencies; it uses the Web Audio API only. Use headphones.

## What it models

- **Rooms**: an 18 × 28 m cinema (~300 seats, stadium rows) with stereo, 5.1 and 7.1 layouts (surround arrays on side and rear walls), and a car cabin with a basic four-door layout or a premium layout (A-pillar tweeters, door woofers, centre, rear doors, sub).
- **Per speaker**: propagation delay (distance / 343 m/s), 1/r level, a simple directivity rule (off-axis level and treble loss), crossover bands, and binaural rendering from its direction relative to your head.
- **Calibration**: time alignment (delay nearer speakers so all wavefronts arrive together) and per-channel level trim, aimed at a chosen point: the cinema reference seat, a car seat, the cabin centre, or "my seat" (follows you).
- **Panned source**: a sound panned across channels with 2D VBAP around the mix reference point, or solo any single speaker.
- **Map overlay**: a predicted image-error, arrival-spread or loudness map. The image-error model is an energy vector with precedence-effect weighting (later arrivals beyond ~1 ms are discounted). It is a teaching model, not a perceptual simulation.
- **Fire wavefronts**: a slowed-down animation of each speaker's wavefront, with the click played in real time.

## Simplifications

Generic browser HRTF (not personalised); direct sound only plus a generic diffuse reverb (no modelled reflections); typical, guessed room and speaker positions; delay and level calibration only (no EQ); one 2.5 kHz crossover; bass under 100 Hz also sent to the sub.
