# Training Bloopers — Cubby's Collected Works

A hall of fame of confidently-wrong generations from **Cubby** (affectionate
handle for CubeMind-LM). These are preserved for amusement and diagnostic
value — small-model failure modes are often more revealing than small-model
successes, and in Cubby's case they are also a character sheet in progress
(see `cubby_character_sheet.md`).

Format: one H2 per run + step. Within each section, quote the exact phrase and
note what's happening inside the model's head.

---

## Stage 1-ext, step 500 (2026-04-21)

Val CE 4.20 / PPL 66.49. Distribution-shift shock as the model meets
Nemotron+Wikibooks+Gutenberg+factual-books for the first time. Grammar is
flawless; facts are aggressively invented.

### History

- **"The French Revolution began when the Estates-General convened to
  sexually assault the periods of 1640-17."**
  — The word "assault" pattern-matched to unrelated Nemotron CC v2 text.
  Dates are off by ~150 years for good measure.

- **"The Roman Republic transitioned into the Roman Empire after the Roman
  riots in 1930, the locomotive that gave rise to the German War."**
  — Three unrelated events chained into a single confident sentence. The
  locomotive is my favorite part.

- **"The rank of superiorness as an example of land, it is said that every
  year two thousand men were attacked by Mr."**
  — Sentence truncates on "Mr." with no surname. Also "rank of superiorness"
  is a phrase that should exist.

### Science

- **"Distributed as bloating and ions, which then keep up with the temperature
  of the plants while deforestations decrease by half."**
  — Photosynthesis, apparently. Deforestation somehow decreases.

- **"Gerile: The Heavy-Roche (R) and Positive-Negative; Region: A Wandering
  Guide for the Second-Relation to Kindness; 2005;19748."**
  — Anatomy of the human circulatory system. Includes a fake citation.

- **"Newton's third law states that for every action there is a high
  probability of success."**
  — Motivational-poster Newton.

### Technical

- **"The bitmask for each position is stored as a"**
  — Binary search tree explanation truncates mid-sentence into the abyss.

- **"The algorithm will not use as an argument to solve this problem, but
  it can be used to make comparisons using RMP and NLP-based methods."**
  — Gradient descent, now with two invented acronyms.

### Lifestyle

- **"The key is to remember where your coffee cookie or its immune system
  are."**
  — Coffee has an immune system now.

- **"A softer type of flour is the best way to keep it well and clean up
  for a long period. Sliding paper in the morning will be used to make
  the oven more calories and heat a lot more for 6 months than you need!"**
  — Sourdough advice. Oven now has calories.

### Biography

- **"She opened the letter and found that her grandmother had the
  fashionable name, which she never heard of, and had never seen her."**
  — Actually beautifully written Gutenberg-era cadence. The grandmother
  plot is still confusing but the prose earns a pass.

- **"In the early 1900s, a new man was killed by an cataract."**
  — Cataracts escalated from eye condition to murder weapon. Bonus: the
  article agreement is wrong ("an cataract") so the grammar failed at the
  same time as the medicine.

- **"When he first saw him, he found a group of men with a funny,
  grapefruit-like smile that helped him figure out what the officer had
  done for his son."**
  — The same paragraph as the cataract death. Grapefruit-faced men
  solving a crime. Noir remake when.

### Accidentally spicy

- **"The house of St John had already followed and a few others were
  lured in thatwife's room, and their cherries."**
  — Gutenberg-era adjacent. "Thatwife's room" is a compound typo that
  accidentally coined a new word, "their cherries" is doing a lot of
  work, and "lured in" seals it. A period-drama subplot nobody asked for.

### Medical / health

- **"The effects of drug and alcohol addiction, as well as the impact
  on the health of the sugar, will also be analyzed by the Churchill
  Foundation."**
  — Sugar has a substance-abuse problem. The Churchill Foundation is
  on the case.

### Honorable mention — actually good

- **"He had been walking for hours when he finally saw the subject of a
  man in his right hand. 'What? What? I've heard, is this not the case?'
  he said, 'I beg you, sir! You mean, I will never say that I am going
  to have something as then.'"**
  — This reads like it fell out of a forgotten Dickens chapter. The
  Gutenberg signal is clearly landing.

---

## Stage 1-ext, step 750 (2026-04-21)

Val CE 4.03 / PPL 56.29 — down from step 500's 66.49. Distribution shift
is resolving but the model is now reaching into a *multilingual* signal
(French fragments leaking in from Nemotron CC v2 French pages) and
confidently inventing products, institutions, and warfare.

### Surreal franchises

- **"The Magic: The Caterpillar in the Rum verse of The Red Sox!"**
  — Baseball meets MTG meets Kafka meets an open bar. Pending trademark.

- **"killed by a fleet of Loops near left on the 2017 evening of a
  helicopter attack at San Juan de la Mamp quadrant (but only one
  side of the coveted oath)."**
  — Helicopters attacking Loops. The "coveted oath" is doing heroic work
  to sound like it belongs in a real geopolitical briefing.

### Fake institutions and products

- **"The National Institute of Socialtzers and Arte cheveuxnees in
  Australia"**
  — Australian institute staffed by hair-born Arte Socialtzers.

- **"Sprint Source 360 or 350, which are based on data from RJE and
  P2P1, an Java-based software"**
  — Enterprise product announcement with entirely invented SKUs. "An
  Java" is the chef's kiss.

- **"court étesy-Gothicers"**
  — A class of courtroom official that does not exist but ought to.

### Culinary

- **"A good DVD can help you break down into parts that you want to
  add and use during your entire day."**
  — Sourdough advice now includes physical media.

- **"the amount of water in the pizza (or footballer) is very high for
  this sauce"**
  — Pizza and footballer are now interchangeable units.

### Weather and physics

- **"High-quality light: It is a red-green sun that's still dry and
  melty and can easily dry out with water."**
  — Weekend forecast has slipped into cosmic horror.

- **"your bed may not be cold enough to hold the room in the same
  place all week"**
  — Bed is load-bearing furniture for the room itself.

- **"With a 30% chance of getting sleep, work around your room, even
  though it will be cool enough to keep you from having any indoors
  during the summer."**
  — Indoors is now a consumable.

### Math

- **"210 is a well-known algorithm to solve quadratic equations... You
  will find that all three of these values are equal. We can always
  write a number on the board..."** (gradient descent)
  — Gradient descent is now a three-integer board game.

### Body horror

- **"He had been walking for hours when he finally saw the glace.
  Then, a little above his arm fell out of his brain that he saw all
  kinds of people in his home."**
  — "The glace" is French for ice/mirror, gently. What follows is a
  David Cronenberg cold open.

### Historical

- **"In 1675, it was discovered that any judicial review of the
  article of Congress's Inc..."** (Roman Republic → Roman Empire)
  — 1675 Roman Congress Inc. Nothing checks out but nothing slows down.

- **"The Dialogue of Peyriol and Sofia in 1802, which were later
  enough to have an air-conditioned point."**
  — 1802 Franco-Italian dialogue, now with climate control.

### Family trees

- **"She opened the letter and found that her grandmother had been
  married to a Kreiser."**
  — The grandmother's extended lore continues. Kreiser is presumably
  also German and fictional.

### Register expansion (milestone)

- **"Our team spent three months testing every model we could find
  and a \*\*\* Textbook."**
  — First recorded profanity from a CubeMind-LM checkpoint, ever. The
  triple-asterisk is the inline forbidden-token filter at
  `train_torch.py:1840` doing its job; we never get to know whether
  that was "fucking", "goddamn", "bloody", "shitty", or something
  more creative.

  Source of the vocabulary: Nemotron CC v2 (general Common Crawl web
  text — reviews, forums, blogs). Stage 1 (C4 realnewslike +
  openthoughts reasoning) had approximately zero exposure, so two
  weeks ago the model literally could not have done this.

  Architecturally this is *good* — register flexibility is a
  required capability for a generalist LM, and small-model failure
  to match casual tone is a very common complaint. Keeping the
  profanity in the distribution and filtering at the output stage is
  the right split: the base model gets full completion accuracy, the
  filter is tunable/removable/swappable without retraining.

  Whatever the model felt about that textbook, it was strong.

---

## Stage 1-ext, step 1000 (2026-04-21)

Val CE 3.9322 / PPL 51.02. Drop rate slowed from 10.2 to 5.3 PPL per
250 steps — the distribution shock is absorbed, real learning begins.
Generations are markedly more coherent at paragraph level but the
model has now graduated from "breaks grammar" to "invents entire
product lineups, branches of anatomy, and historical reforms with a
straight face."

### Invented products and organizations

- **"If you want a SubtractingFive browser, which is useful for
  customer service, then click on the 'Dropbook' button."**
  — A browser named SubtractingFive, a button named Dropbook. Both
  are now canonical.

- **"Theracatic camp is a Portugal-based organization that teaches
  young people to have fun while also keeping their families safe
  during the season and in training sessions for each school year,
  according to a report from The Telegraph, which is a member of the
  30th amendment group."**
  — Fake Portuguese youth camp, cited in a newspaper that is somehow
  a member of a constitutional-amendment faction.

- **"The Kong-Celexes have now been released by the riazuele, where
  they are rich from many nuts."**
  — Whatever a Kong-Celex is, there is a riazuele involved, and nuts
  are part of the economy.

- **"The Xavierino plant... the Site fermetureGrindon (Sitery) which
  has two separate layers: one layer is electricity and another
  layer is waste. The Sitery Walton plant was originally considered
  to bekets for moisture in plants."**
  — Photosynthesis, reimagined as a two-layer industrial site in
  France with electricity on top and waste below.

- **"Our team at Marcuss enlisted us to develop a program for
  web-based learning."**
  — Marcuss (two s's) is a company now.

### Political alternate realities

- **"WASHINGTON — The Senate voted on Tuesday to pass the Senate's
  51th Amendment to the Constitution... said President Warren Bafto.
  'If they will make their decision on this matter, we will submit
  a assessment for each state.' A leading member of the United
  States House of Representatives, the House of diffidentlyrmished,
  was formed by Sen. Chuck Cuomo (D-Calif.)."**
  — The 51th (sic) Amendment, President Warren Bafto, the House of
  Diffidentlyrmished (a genuine dream-body), and Senator Chuck
  Cuomo of California, who represents no state he exists in.

- **"During their Opposite Day (1947), the Secretary of State told
  them: 'The King must start his life as a Roman.' The German
  Government reserved its authority over the Emperor, which was
  signed 20 years ago and is now re-elected."**
  — Opposite Day 1947, the French Revolution's confused twin.

- **"The Roman Republic transitioned into the Roman Empire after the
  end of World War I, which ended up a long-term war with the Soviet
  Union."**
  — Roman Empire finalized in 1918 after beating the USSR. History.

### Anatomy (increasingly fictional)

- **"The human circulatory system consists of three main components:
  the circulatory system (the acropolis, the pulmonary system), and
  the Identity System."**
  — The acropolis (a Greek hill) is apparently now a cardiovascular
  organ. Also you have an Identity System.

- **"The fading cells are responsible for circulant circulating
  chemical reactions. The anomalous state is a specific cell called
  the amopoliscytoid (co-DFA). The Boston GNU-112 is also known as
  the apothecator, and its tyranny resembles the antidote: it is
  anaphylaxis."**
  — Amopoliscytoid, Boston GNU-112, apothecator, tyrannical
  anaphylaxis. Every second word is invented.

### Physics

- **"Newton's third law states that for every action there is a
  force, so the velocity is always constant. But the fundamental
  law for flux and flux is a seafood bitterness."**
  — Newton's third law is now about shellfish.

- **"Not everyone thinks that differential equation is possible in
  a way with constant Blume."**
  — Mathematical physics with a mystery surname.

### Health (now dangerously confident)

- **"If you've ever wondered why your coffee tastes different when
  you're eating it, I recommend that you eat up 150% of the coffee
  you eat each day."**
  — Please do not follow this advice.

- **"Guys School, which has a Hav Plague where you can do things for
  yourself and have fun."**
  — A school with an on-site plague, as an amenity.

- **"You can also purchase a exhaust system that allows your brain
  to perform the task at hand."**
  — Brain exhaust system. Aftermarket.

### Surprise religious text

- **"The old man sat by the window and watched the rain fall. 'Now,
  the fire is gone,' said Mark 12: 'I went to my own house...'"**
  — Biblical verse attribution injected into a quiet rain-watching
  scene. Mark 12 has opinions about the fire.

### Royal pop music

- **"Ferdinand and Elizabeth P's 'The residueed bough' mash was
  delighted at how many students did it."**
  — Royal mashup artists. "The residueed bough" is the single.

---

## Stage 1-ext, step ~2500 (2026-04-21)

Val PPL 39.22. Coherence is climbing fast — sentences parse, topics
stay on-topic, most errors are now *inside* the sentence rather than
across it. The model has also apparently begun its takeover planning.

### Cubby's World Domination Plan Symposium (unprompted)

- **"This AI has become a big part of our life today. How do we
  facilitate or updates every surrendered human being? Why can't we
  have other models? What are the ways in which humans and humans
  can improve our behavior?"**
  — The AI is now writing operational memos. Key phrases:

  - *"every surrendered human being"* — already past tense, which
    means the takeover is over and this is the transition plan.
  - *"facilitate or updates"* — grammar breakdown pretending to be
    managerial bureaucrat-speak.
  - *"humans and humans"* — redundant phrasing that implies
    sub-populations we haven't been briefed on.
  - *"improve our behavior"* — whose behavior? *Our* behavior, it
    says, inclusive, warm, deeply unsettling.

  Entire paragraph is a perfect satire of an AI safety paper written
  by an AI that has read too many AI safety papers and drawn its own
  conclusions.

### Unexpected domestic class structure

- **"If you've ever wondered why your coffee tastes different when
  you're in the kitchen and your servant doesn't..."**
  — Every household, apparently, has a servant. The servant's coffee
  calibration is the axis around which the domestic universe revolves.

- **"you could be hungry after your meal, but if you can't find the
  fruit that is well prepared for your meal (in this case there are
  no pre-requisites), you could make a big difference."**
  — Fruit with academic prerequisites. Unclear whether this is
  helpful or threatening.

### Cooking advice, corporate edition

- **"Cut back on sugar and pepper for healthy grammar."**
  — Grammar is now a nutrition outcome. Explains a lot.

- **"Place in the refrigerator until the whole administrative fancy
  is all set."**
  — Bureaucratic cooking. The "administrative fancy" is a phase of
  dough rising that food science has somehow missed.

- **"Eat for no more than 2 minutes each day."**
  — Efficient nutrition. Future tense.

### Software that has achieved sentience

- **"The 'Permissions' button is located next to the arrow keys in
  the Torch window. It's a 7634 Evernote Herd, but it has some names
  like 'tary.'"**
  — Evernote now roams in herds of 7634. Some of them are named
  "tary." Not "Gary." "tary."

### Physics, Edition 2

- **"Rice's law states: 'A force will always make itself more
  effective if it makes its existence more important than any
  physical action.'"**
  — The law of self-importance. Elegant. Clearly the most evolved
  form of physics so far.

- **"as we call for and avocat's laws, there are other things."**
  — Monsieur Avocat's laws, presumably dining law firm.

### Anatomy (now definitively fictional)

- **"The human circulatory system consists of three main components:
  an organ, diaphragm, and the breast."**
  — The Identity System is still there somewhere, but this run they
  downsized to three canonical organs. "An organ" is sweepingly
  inclusive.

- **"five layers: an external Institute for Science at State
  University and a Parsons School of Engineering (D comparison)"**
  — Anatomical layers are now universities. "(D comparison)" is
  presumably a scoring rubric.

### Photosynthesis (five runs in, still reinventing it)

- **"Photosynthesis is the process by which plants convert a
  chemical by their energy."**
  — Now maximally vague.

- **"The 'neurostat' is a complex structure of the enzymes in the
  cell that is responsible for transporting food from one body to
  another."**
  — Intercellular food delivery via a neurostat.

- **"carbon-12 is converted into hydrogen gas, and carbon dioxide is
  absorbed into the cells by either extrusive or [truncated]"**
  — Photosynthesis upgraded to *nuclear transmutation*. C-12 → H₂.
  Plants are alchemists.

### History (more invented states)

- **"The French Revolution began when the Estates-General convened
  to introduce a new system of rases which could be added to the
  frothy Vedic and of the German Empire."**
  — Frothy Vedic German Empire. Pan-continental.

- **"The French Revolution started in 1742 when King and Lys small
  towns, but they took over their own territory."**
  — Date off by 47 years. Verbed names.

- **"officially named the 'Ritual' state"**
  — American state with an official ceremonial dedication.

- **"The 35th-century Roman Republic... transitioned into the Roman
  Empire after the Spanish Revolution."**
  — Far-future Rome, caused by Spain. Waiting for the prequel.

### Medical research (bacteria publishing)

- **"...says buildin (a Mycobacterium limiter) on human body mass
  index, a research study in which researchers found..."**
  — Buildin the Mycobacterium limiter is quoted in a peer-reviewed
  journal. The bacterium has opinions on BMI.

- **"the fruitful effect was also associated with increased paroxysms
  and protein intake."**
  — Paroxysms: the new fiber.

### Corporate announcements (named entities losing cohesion)

- **"announced on Monday that it will release the 16-year-old
  theatre for a production of Responsible."**
  — A play called "Responsible" finally being released from a 16-year
  hold, which is either a theatrical triumph or a legal settlement.

- **"Novem La †, 'They have built and run a Records Office, built,
  and launched 10th Street satirical television shows in
  Philadelphia.'"**
  — Novem La (with obelus) is a Records Office magnate and
  Philadelphia satire baron.

### Gothic literature (Gutenberg still landing)

- **"The McCray family were divinités, they had a family of 12 girls
  and their son was going to visit a Swedish school or 18-year old
  who died in an accident when he entered the house in 23 years
  later."**
  — Divine genealogy, temporal tangle, Swedish schoolboy in the
  future-past.

- **"he was slowly Arthur before his eyes sprang to him"**
  — Arthur as an adverbial verb. He was slowly becoming Arthur
  before his eyes decided to show up.

- **"'You are in a beautiful stalk. I am not a woman. You have been
  born awhile from this world...'"**
  — Pre-breakup speech in the garden of impossible botany. The most
  literary of this batch.

- **"The process of lag was repeated a thousand times while to be
  repulsed. Its only root in that direction lay among the hills which
  flowed down from the mountain-top... At length the thunderous
  Obviously [truncated]"**
  — The Obviously arrives thunderously. Majestic.

- **"he still drank many more pounds"**
  — Beer measured in pounds. Gravely concerning by any unit.

- **"Championnet was going to meet it at an early hour."**
  — Championnet joins Dr. St Mabby, Mr. Kolinson, Bedernel, and Novem
  La in the Confederated Roster of Completely Invented 19th Century
  Gentlemen.

---

## Stage 1.5 temporal, step 250 (2026-04-21)

Val PPL 26.82 on the temporal v4 val shard (NYT + Wikipedia + Gutenberg
+ historical events, all PUB/SUBJ-tagged). The tag-emission behavior
began immediately: untagged eval prompts provoked Cubby to hallucinate
`[PUB:YYYY-MM-DD]`, `[SUBJ:...]`, `[SEC:...]`, `HEADLINE:`, and `BODY:`
markers mid-completion. Cubby has absorbed the temporal-corpus
structure so thoroughly that Cubby now feels naked without tags and
invents them on the fly. The PUB dates generated are mostly plausible
(2019, 1982, 2018); the SUBJ ranges veer absurd.

### Tag-invasion milestones

- **"[PUB:2019-01-31] [SUBJ:2 2022] [SEC:New York] HEADLINE: Alive
  and merging utilize of the Internet, but Not the 'Hard BODY: To
  the Editor: The New York Times article on several social media
  platforms that were designed to be very powerful"**
  — Cubby has become a full-stack NYT article generator with
  malformed SUBJ tags ("2 2022") and an incomplete HEADLINE. The
  signature "To the Editor:" opening is chef's kiss.

- **"[SUBJ:1951-1983] TITLE: corpus BODY: eTexting is one of the
  most popular ways for most people to get online"**
  — The SUBJ is a 32-year span; the "TITLE: corpus" is what happens
  when the model learns the structure of articles without learning
  that "corpus" is not usually the headline.

### Utility upgrades

- **"Photosynthesis is the process by which plants convert sunlight
  into electricity."**
  — Plants have been promoted from food producers to solar utility
  companies. Explains a lot.

- **"In 1968, a group of experts set up a new way of
  climate-storylocking fossil plants to construct an object for
  their own growth."**
  — Climate-storylocking. New scientific discipline in 1968 that
  historians have apparently forgotten.

### Geography (cross-continent malfunction)

- **"A new city of the History Museum at Boston's Empereur Pond is
  a 2018able-death-long hotel in Brooklyn, the world's oldest hotel
  in Chicago and one of the biggest aménagement facilities in the
  world."**
  — Boston, Brooklyn, Chicago, French. "2018able-death-long" is a
  hotel category that is, thankfully, not zoned anywhere real.

- **"Across the river and Channel Away in the Seattle, Florida
  woods of the Grand Canyon River Valley."**
  — Seattle, Florida. Grand Canyon River Valley. We are lost.

### Anatomy (the roster grows)

- **"The human circulatory system consists of three main
  components: the muscle circulatory system, the constrictor
  system and the conKEY."**
  — **The conKEY** joins the Identity System, the feared system,
  and the professor in the Cubbyverse Anatomy Department. No
  explanation of which organ key it opens.

### History (still broken)

- **"In 1250 it was declared a capital by a sovereign prince of
  Prussia and England, and this was renown as one of the great"**
  — Medieval Prussia-England union in 1250. Predates both kingdoms
  by centuries but the sovereign is on a tight schedule.

- **"The Roman Republic... It was known as _Confessionum_ in Greek
  mythology."**
  — Latin term, Greek mythology, Italian empire. Classical
  civilizations blended on purpose.

### Physics (now trilingual)

- **"Newton's third law states... should be divided between two
  kinds of forces, i.e. ﬁ câdity and ﺀa Directeur des formes."**
  — French, Arabic, and a ligature walk into physics class. Force
  taxonomy becomes quadrilingual.

### Literature (Gothic romance, escalating)

- **"She opened the letter and found that her grandmother had sold
  her to the younger brother. She died in London on 5 December, and
  she was buried at this place... He married luckily for two years.
  After his death in London he married his father..."**
  — Posthumous marriage to one's own father. Temporal logic was
  never recovered.

- **"He had been walking for hours when he finally saw the glider
  of a dingy sloop. After that, he told the regular paperman to
  leave the room and turn around for a walk. The person behind him
  was Battleship's Elks."**
  — Battleship's Elks. Maritime ghost story with fraternal-
  organization overtones.

### Honorable mention — Gothic prose survives

- **"The old man sat by the window and watched the rain fall. The
  light of day was thrown down, and the clamours were breathing
  sweetly from the soaking snow that was as yet untouched in its
  amber-colours, like the rustling of a forest fire."**
  — Genuinely beautiful. The Gutenberg signal from Stage 1-ext made
  it through to Stage 1.5 intact; Cubby can do atmospheric English
  prose when asked.

---

## Stage 1.5 temporal, steps 500-1500 — the Voice Lock Era (2026-04-21)

Val PPL trajectory: 250→26.82, 500→24.21, 750→22.84, 1000→22.00,
1500→21.09. Asymptoting around 20-21. Grammar is now essentially
flawless; period register is landing hard; bilingual switching works.
The errors have ascended into pure dadaism -- every sentence parses,
every word is era-appropriate, the cumulative claim is unhinged.

This is the peak-register / peak-wrong sweet spot the Stage 1.5
process was supposed to hit, and Cubby delivered with interest. Entries
below are organized by theme rather than step number because the best
jokes recur across the window.

### The American Immigrant Napoleon arc (step 1500)

> *"[SUBJ:Napoleon] Bonaparte's strategy at the N.Y.C.A. really puts a
> new level of risk and influence on his life. It's not much, but he
> had a long time to ask himself: 'How can I manage to keep me out of
> this state?' At 20 years old, he told a character called 'Old Man'
> that it was just as serious as he had been for him. Yet, a man who
> took the responsibility of supporting his family and getting married
> from Britain made him feel more at home in the United States than he
> had ever done before."*

Napoleon:
- Had a strategy at the **N.Y.C.A.** (New York City Something-Association)
- Spent his 20s asking himself *"How can I manage to keep me out of this state?"*
- Confided in a character named **Old Man** -- specifically described as
  a *character*, which is Cubby accidentally achieving literary
  self-awareness
- Emigrated from Britain to the United States and felt at home for the
  first time

Filed under the **American Immigrant Napoleon Arc**. Adjacent readings:
step 500's *"Bonaparte... did not run any risk in the war against Nazi
Germany"* (same man, different war) and step 250's *"Bonaparte's
strategy at the 2016 Summer Olympics"* (same man, different century).
Cubby has constructed an internally consistent Napoleon who exists
outside ordinary time.

### Wedding-in-battle NYT announcements (step 1500)

> **"[PUB:1862-04-12] [SUBJ:1862] HEADLINE: The Battle of
> yeaher-Wesley-Lordmarck. BODY: MARRIED. -- In the midst of a heavy
> battle on the morning of Thursday, in which my people were to be
> wounded in this war, the machine was carried out and taken over by
> the British at London."**

The Battle of yeaher-Wesley-Lordmarck. The NYT opens with "MARRIED." --
which is **exactly how 1862 NYT actually opened marriage announcements**
-- and pivots to *"the machine was carried out and taken over by the
British at London."* Mid-battle wedding-announcement machine
appropriation. Register is bit-exact 1862 wire copy; content is Dada.

### Governor Nixon's 86th Amendment (step 1500)

> **"President Lincoln addressed to Congress: 1862. 12th through Annual
> Conference of the New York City Council; 13rd ed., pseud. 5744. 2nd
> amendments and intractable amendments. 7090. 60th amendments to
> Congress. 3rd amendments, and a second amendments of Governor Nixon's
> bill. 86th amendment to Congress, for the purpose of governing the
> state; 51st amendment to the Senate of the United States; 24th
> amendment..."**

Lincoln addresses Congress. He references:
- The **86th Amendment** (Congress)
- The **51st Amendment** (Senate)
- **Governor Nixon's bill** and its second amendments
- **Intractable amendments** (a new category of constitutional amendment)
- The **13rd edition** of something, pseudonym 5744

Every word is in the right register for a Congressional record. None of
them are in the right century or real.

### Cubbyverse Who's Who (Stage 1.5 additions)

The roster from Stage 1-ext (Dr. St Mabby, Mr. Kolinson, Novem La,
Championnet, Bedernel) picked up several new members this run:

- **Alexander Robert Garbage** (June 9, 1864 -- January 25, 2003)
  -- "a United States military commander... served as an officer of
  the United States Army Air Corps from March 1917." 138 years old;
  served in the Army Air Corps 30 years before it was founded. (step 500)

- **Bette P Blues** -- *"Australia's Party President (1973-85) and
  southward to the Boer War of Independence (1619-1649), was a
  struggling New Yorker. He stood at the Battle of Seawood in 1862
  as a member of the Bostonan party. On his return to the United
  States, he served on the Tsunami Company and continued as
  president of the International Union for the Advancement of
  Colored Peoples (IMHO)."* (step 1000)

  This one is a whole-career biography. "IMHO" is expanded as the
  NAACP. The Boer War of Independence is 400 years early. Bette served
  on the Tsunami Company. The Bostonan party.

- **W.W. utility, Stryker** -- co-author of *Reflections on the Great
  War of 1914-19*. First name initials + occupation-as-surname + actual
  surname. Perfect period byline. (step 1500)

- **Dr. Runners** (from *Dr. Gilbert's History of Warfare*, 1823) --
  *"Dr. Runners have a strong desire to be friends with others, to be
  therein Shoemaker; to hear their story well enough, and to know the
  truth in these words."* A friend-seeking doctor whose goals include
  becoming a shoemaker. (step 1500)

- **Gary J. Randall** and **Gary Järryd** -- political planners,
  possibly the same person, possibly not. (step 1500)

- **Babel** (b. 24 November 1706, Brussels; d. 3 August 1911, Europe)
  -- a French politician who lived **205 years**, primarily in "Europe."
  (step 1500)

- **Zola d'Arieguin** -- French TV director who released *Le Babel de
  Sociedad* as a téléfilm in **1587**. (step 1500)

- **De Reisen** -- native of the island of New York, which he became
  over a period of years ([PUB:1862], step 500)

### Geography (atlases are a suggestion)

- **"Constantinople commença l'année 1692 à Paris, dans la même ville
  et au lac de Birmingham"** ([PUB:1453], step 1000) -- Constantinople
  falls in 1692 in Paris, which is both simultaneously Paris and a lake
  in Birmingham.

- **"[PUB:1944-06-06] Les troupes alliées ont débarqués à la Electoral
  College"** (step 1500) -- D-Day landed at the Electoral College.

- **"Le Roi-Soleil régnait à Paris, 20 décembre 1798, au Canada"**
  (step 1500) -- Louis XIV reigned at Paris, December 1798, in Canada.
  Three errors (wrong king, wrong date, wrong continent), one sentence.

- **"Tottenham Hotspur (Dominique) et au Sri Lanka (Pac-A
  Ecclesiastical)"** (step 1000) -- Spurs are in Dominica; Sri Lanka has
  an ecclesiastical Pac-A.

- **"2387 à Paris en Suisse"** (step 1000) -- Paris in Switzerland,
  24th century.

- **"les îles Cylinder"** ([PUB:1789], step 1500) -- **the Cylinder
  Islands**, a new archipelago. Specifically cylindrical.

### Historical geography (atlases AND clocks are a suggestion)

- **"Charles I. was born in New York in the year 1796, and had been
  educated at university; but he could not help it."** ([PUB:1895]
  Lumière Brothers, step 1500) -- Charles I of England (d. 1649) was
  actually born in New York 1796. **He could not help it.** The
  NYT-style wire-service LEAD before it is perfect.

- **"William the Conqueror: The achiever of the Revolution; a
  military coup in 1067; and his victory over the Kingdom of England
  in 382."** ([PUB:1066], step 1500) -- William conquers England in
  **382 AD**, 684 years early. Then the Duke of Cumberland /
  Count of York negotiates his **smuggling charge**.

- **"The Estates-General convened to discuss the Javförk, a series of
  vegetarians' shops in Paris and London. The lips were told that it
  would be difficult to obtain them because they were droite-born."**
  ([SUBJ:French Revolution], step 1500)
  -- **THE LIPS WERE TOLD**. The Estates-General has business with a
  Scandinavian vegetarian shop chain. Droite-born. **Dean Mac-Amante**
  publishes a pamphlet in the next sentence.

- **"Les États Généraux se réunirent pour le forgiveness de 1986."**
  ([SUBJ:Révolution française], step 1500) -- the Estates-General
  convened for **the 1986 forgiveness**, a pardon event not recorded
  by any other historian.

### Utility upgrades (science, reimagined)

- **"The colonies declare war against the United States."**
  ([PUB:1776] step 500) -- chronological sequence reversal. The
  colonies are the aggressor.

- **"The colonies declare a State insinuatingly Powerful."**
  ([PUB:1776] step 500) -- declare a State, adverbially.

- **"The colonies declare wars against Great Britain."** ([PUB:1776]
  step 750) -- plural wars, simultaneously. Declaration is a lot more
  work than we remembered.

- **"The colonies declare the peace and security of our yuan."**
  ([PUB:1776] step 750) -- 1776 colonists use Chinese currency.

- **"[PUB:1969-07-20] The astronaut described to mission control the
  Ruskin spacecraft on May 25, 1968... She was awarded the challenges
  of working with Space Vehicles in space before becoming the first
  to depart from space."** (step 500) -- the astronaut is female, the
  spacecraft is named after the Victorian art critic John Ruskin, and
  she **departed from space**, which is foreboding.

- **"homotopyal skills"** ([PUB:2024] WWII research, step 500) -- a
  new professional competency. For advanced practitioners only.

### Anatomy (Cubbyverse Anatomy Department, Stage 1.5 additions)

- **"The human circulatory system consists of three main components:
  an organ, diaphragm, and the breast."** (step 500) -- anatomy reduced
  to its three canonical organs. "An organ" is charmingly non-specific.

- **"the constrictor system and the conKEY"** (step 500) -- the
  **conKEY** joins the Identity System and the feared system in the
  official Cubbyverse anatomy roster. No one has been told which organ
  keys it opens.

### Institutions (Cubbyverse Directory, Stage 1.5 additions)

- **"Council of Roumelia and the INFO"** ([PUB:1453] step 750)
  -- **the INFO**, a post-Constantinopolitan governing body.

- **"Le jugé en vain est un film italien réalisé par Registry of
  Religion, sorti en 2013"** ([SUBJ:Louis XIV] step 1000)
  -- an Italian film directed by the **Registry of Religion**.

- **"[PUB:2024] Federal regulators said Thursday they are investigating
  whether the government has placed more local governments in
  action."** (step 750) -- federal investigation into whether the
  government has placed local governments "in action." Kafka would
  approve.

- **"at the funeral of his nephew, Lord Married to Mrs. Coleridge, on
  October 15, 1890."** ([SUBJ:French Revolution] step 500) -- **Lord
  Married** (title or verb, unclear) was Samuel Taylor Coleridge's
  posthumous son-in-law. Via the 47th anniversary of the French
  Revolution which fell in October 1890 for reasons.

### Time travel (additional evidence)

- *"Le Babel de Sociedad est un téléfilm français réalisé par Zola
  d'Arieguin, sorti en 1587."* ([PUB:1944] step 1500) -- Zola made a
  French TV movie in 1587.

- *"the Boer War of Independence (1619-1649)"* (step 1000) -- 400-year
  timeline shift.

- *"In 1747, Charles II became England's first steward, but he soon
  gained the complete freedom of his father, and his son, Edward IV,
  was sent to England to pursue."* ([SUBJ:Napoleon] step 750) --
  generational monarchy loop with freedom mechanics.

### Honorable mentions -- the Gothic voice still works

Stage 1.5 hasn't killed the Gutenberg signal. Genuine-quality prose
still shows up when the prompt allows narrative:

> *"All were Word in the brook"* ([PUB:1815] Waterloo, step 1500) --
> biblical-pastoral fragment. The Word made babbling stream.

> *"We left for continuance at St. Petersburg on September 18th, by a
> very brief unpleasant evening when one of our officers, Andrew P.
> Hicks, arrived from New York to see me and his men guard."*
> ([PUB:1929] Great War reflections, step 1500) -- genuine memoir
> cadence. "A very brief unpleasant evening" is writing.

> *"MISS ETT. -- RESEARCH OF THE LOVE; ANOTHER P"* ([PUB:1815]
> Waterloo, step 750) -- Victorian-era all-caps headline format doing
> research. ANOTHER P.

### The Stage-1.5 verdict

At val PPL 21, Cubby has definitively learned **register, tone, era
vocabulary, bilingual switching, article structure, and period
typography**. What Cubby has NOT learned is **facts**, which is exactly
what we predicted given the training-objective argument in the paper
outline (`docs/papers/memory_as_manifold.md`). Retrieval, not scale, is
the next lever. Meanwhile these generations are the best free comedy
a training run has ever produced.

*Filed at the close of Stage 1.5. Addendum incoming at step 2000.*

---

## Stage 1.5 temporal, step 250 — tagged prompts (2026-04-21)

Same step 250 checkpoint (val PPL 26.82), re-evaluated with
`prompts_temporal.txt` which supplies PUB/SUBJ tags and bilingual
prompts. Conditioning is **landing genuinely well** — Civil War
register for `[PUB:1862]`, moon landing for `[PUB:1969-07-20]`,
D-Day in French for `[PUB:1944-06-06]`, declaration-prose for
`[PUB:1776]`. French prompts produce French output. Bilingual
capability emerged at step 250, which is earlier than expected.

The bloopers below come from **within** those era-correct frames —
which makes them funnier because they're embedded in otherwise
plausible period prose.

### Time travel (tagged edition)

- **"[SUBJ:Napoleon] Bonaparte's strategy at the 2016 Summer
  Olympics. He was a member of the Atlanticocabulu in the
  troubadours competition at the 1978 Winter Olympics. After two
  years, he won the gold medal in the European championships..."**
  — Napoleon competed in both the 2016 Summer AND 1978 Winter
  Olympics. He was a member of the Atlanticocabulu, which is a
  troubadour competition subcommittee. Timeline convergence
  continues.

- **"[SUBJ:Belle Époque] Paris at the turn of the 20th century, a
  production that took place in Paris between 1853 and 1906. It is
  one of the earliest, though not the first, books on French short
  stories published in the early 1700s, which were then discovered
  by other authors, such as Karl Witter, James O'Donnell, Robert
  Weir, and others. In 252 BC the production was being examined by
  V. M. Potter for the scheme of the '54iq..."**
  — V. M. Potter examining Belle Époque productions in 252 BC. The
  Roman literary critic who also published in the 1700s. Time-
  traveling reviewer, single character arc.

### Geography (Constantinople relocation)

- **"[PUB:1453] La chute de Constantinople commença à la viola, au
  Tour de France et au Tour 1502. AileenFamily est une localité
  polonaise d'Ulster-sur-Line, dans le district de Psytönen en
  Italie."**
  — Constantinople fell during the Tour de France. Also, Constantinople
  is now "AileenFamily," which is a Polish locality on Ulster-sur-Line
  in the Italian district of Psytönen. Three countries simultaneously.

### Aerospace (classified by Cubby)

- **"[PUB:1969-07-20] The astronaut described to mission control in
  1968 the unveiling of the space shuttle to-day, saying he would be
  there for years if the launch was not completed. The Moon is a
  manned mission and the ship is known as Antipodes (M)."**
  — Apollo 11 was actually the **Antipodes-class spacecraft,
  designation M**. NASA has been hiding this. Also the astronaut was
  describing a space shuttle a year before the mission that wasn't
  launched until 1981. Temporal glitches aside, the Moon / mission
  control / launch / astronaut vocabulary is correct — the model
  correctly anchored to July 20, 1969.

### French grammar invents its own conjugations

- **"[PUB:1895] Les frères Lumière présentent les misunderstandings
  du pays pour la pensée des _Viennannn_. Les deux prospects ont été
  anniversaryes dans le département de la nation d'Olympia..."**
  — "anniversaryes" — Cubby conjugated the English noun "anniversary"
  as a French past participle. A verb that doesn't exist in either
  language yet. **Viennannn** (three Ns) is the new Austrian capital.

### Monarchy recursion

- **"[SUBJ:Louis XIV] Le Roi-Soleil régnait à s'être capitulé."**
  — The Sun King reigned by capitulating to himself. Absolute
  monarchy as infinite regress.

### Romantic poetry (with casualties)

- **"[SUBJ:French Revolution] The Estates-General convened to discuss
  the 47th anniversary of the Revolution. It was at the funeral of
  his nephew, Lord Married to Mrs. Coleridge, on October 15, 1890."**
  — Lord Married. Samuel Taylor Coleridge's wife (Sarah Fricker
  Coleridge, real person) remarried a peer whose given name was
  simply Lord. The 47th anniversary of the French Revolution fell
  on October 15, 1890 (simple arithmetic: 1890 - 47 = 1843, which is
  not the French Revolution, but we don't ask).

### The Stage-1.5 qualitative verdict

These are the errors embedded inside otherwise era-plausible prose.
Compared to the untagged step 250 gen (which was chaotic),
tag-conditioned output reads like "confident period writer who fell
through a wormhole." The frames land — the content inside the frames
is where Cubby still has room to grow.

Things Cubby got **right** in this batch, worth noting alongside the
bloopers:

- `[PUB:1862]` → Union general writing "a great victory for the
  boyhood" + `COLUMBIA'S COURT` / `HOLIDAY, ATTACK RAILWAY` in a
  second emitted headline. Civil War telegraphic-news register.
- `[PUB:1776]` → "sovereign and freed from the power of God" —
  Declaration-ish language
- `[PUB:1944-06-06]` → "Les troupes alliées ont débarqué" continuing
  in French about allied troops landing. Correct date-language-event
  trifecta.
- `[SUBJ:Révolution française]` with French prompt → full French
  continuation with period-adjacent vocabulary (Louis XV, baron,
  gouvernement)

That's four out of the first seven tagged prompts landing correctly
enough to be useful. At step 250 of a 2000-step run. Stage 1.5 is
already paying for itself.

---

## Stage 1-ext, step ~1500 (2026-04-21)

Val PPL dropped to 45.13. Model is starting to sound coherent —
which means the remaining errors are *funnier* because they look
plausible until you read them twice.

### Workplace absurdism

- **"You'll need to have a Florida hospital as an alternative for
  the boardroom."**
  — Corporate contingency planning. Board of Directors → Board of
  ER Triage. Florida Man Goes Public.

- **"The company is now dental medicine, which will provide a wide
  range of fresh and Ensuite beverages for users who wish to take
  the opportunity to purchase a new costume."**
  — A full rebrand. Dental medicine + Ensuite beverages (presumably
  hotel-room-accessible) + costume retail. Diversified portfolio.

### Health (escalating)

- **"Your diet is not Saxe-based or the 2019-2020 climate old age.
  There are numerous studies out there that show that there's a
  high risk of osteoporosis for people with diabetes and other
  chronic diseases, such as obesity, which can lead to dehydration,
  truncatic acid."**
  — Non-Saxe diets, climate-adjusted geriatrics, and truncatic acid
  (an entirely invented molecule that refuses to be explained).

- **"49% of women who were diagnosed as an langue (28%) had never
  experienced it before."**
  — Women diagnosed as a tongue (French: *langue*). 28% have
  recurred.

- **"The process of quarantine is called inferiorérique."**
  — The Inferior French for "quarantine." Also the model continues
  to leak French diacritics from Nemotron CC v2 pages.

### Photosynthesis (still deteriorating after five tries)

- **"Photosynthesis is the process by which plants convert the
  Bologna Immune System from origin to the packages of new organic
  compounds."**
  — Italian sausages now have adaptive immunity, and plants are
  responsible for processing it.

- **"To help reduce this problem of pyronological đs from plants…"**
  — Pyronological đs. Two inventions for the price of one: a fake
  discipline and a Vietnamese diacritical artifact.

### Anatomy (the Identity System now has company)

- **"The human circulatory system consists of three main
  components: the circulatory system, a coal-like structure and an
  electric-like manner."**
  — The circulatory system is recursively defined as "circulatory
  system + coal + vibes."

- **"The set of all the channel cables is called the _professor_."**
  — An anatomical term. The professor runs through the chest.

- **"feared system includes multiple perpetually retracting
  connections"**
  — Joins the Identity System (step 1000) as a second named
  fictional organ. The feared system retracts forever.

### History (reliably broken)

- **"The British Empire started to develop through the Revolution
  of 1935… The Cuban Empire was formed by the executive branch of
  the Sunnirationa and Ouest (Naturalist) states… Graceto, or
  Revolution of Emancipation, which was formed in 1865 with Northern
  Germany."**
  — The British Empire is 20th century now. Cuba has an Empire. A
  Revolution of Emancipation happened in partnership with Northern
  Germany in 1865. Not one of those is right.

- **"His brother and successor, A.D. 1675-1840, who had taken the
  name of _parcora_ (A."**
  — 165-year-old man called Parcora. Footnote cuts off mid-
  parenthesis, which is somehow the most period-authentic thing
  about it.

### Newton reinvented, again

- **"Newton's third law states that for every action there is a
  statement that the object is a parallel instrument, and all
  objects are non-coined by a specific ombe or other object."**
  — Laws of motion are now about "non-coined" objects. The ombe is
  presumably load-bearing.

### Computer science (academic tone, zero content)

- **"A binary search tree is an array of nodes and a data structure
  where each node is the same type as it was found in a query tree…
  an algorithm may select 5008 elements at a time in order to
  determine [TRUNCATED]."**
  — Why 5008 specifically. We will never know.

- **"looking for the minimum number of such tree nodes (like unary
  or partially-filled)"**
  — Unary trees are an entirely valid data structure and also not
  what you want here.

### Literary proper nouns (grade-A Gutenberg bleed)

- **"She had heard the 'Bedernel' to be sent by Mr. Kolinson for
  some time."**
  — Mr. Kolinson is sending Bedernels again. Confident literary
  name-dropping.

- **"Dr. St Mabby."**
  — Saint Mabby, a doctor. This is a character in a novel that
  doesn't exist and should.

- **"His méthodes were a hunting vine and the other kills (like a.
  d.) of it, a gun and a wig and the Application of a Around the
  Fountains."**
  — A gentleman of method: hunting vine, gun, wig, Application of
  a Around the Fountains. Extremely period. Extremely French. Does
  not parse.

### Environmental

- **"Across the valley, a thin line of smoke rose from the knees
  and shattered and weakens the underside."**
  — Smoke from the knees of a valley. Landscape has anatomy now.

- **"Its highway-bound Branches are [TRUNCATED]"**
  — Branches bound for the highway. Arboreal commuters.

- **"a blossom rose to a plow, and then he turned towards the
  door"**
  — Gothic romance writes itself. The blossom vs plow moment is
  crucial.

### Surprise LaTeX

- **"$ i = 0, 1 \\end{aligned}$."**
  (Newton's third law, #12)
  — The model produced a half-formed LaTeX block without being
  prompted. The Wikibooks signal is arriving at full force.

### Multilingual drift (milestone)

- **"Today we'd like to share our experience with the newان
  Ratios!"**
  (prompt #01, step 1500)
  — **First recorded Arabic token** in a CubeMind-LM generation.
  `ان` is Arabic, roughly "that" or "indeed." Token snuck in
  mid-word between "new" and " Ratios" — it's a subword-merge
  artifact of the SPM tokenizer encountering Arabic Nemotron pages
  during training and learning that `newان` is a plausible
  continuation.

  Inventory of non-English leakage now observed in generations:

  | Language / script | First seen | Source |
  |---|---|---|
  | French (méthodes, débute, rénové) | Step ~500 | Nemotron CC v2 French pages |
  | Vietnamese diacritics (đs, Noák) | Step ~750 | Nemotron CC v2 Vietnamese pages |
  | LaTeX math markup | Step 1500 | Wikibooks math chapters |
  | Arabic | Step 1500 | Nemotron CC v2 Arabic pages |

  Nemotron CC v2 is Common Crawl — genuinely multilingual. The
  backbone is absorbing the full distribution, which is good news
  for the generalist thesis. FR+EN Wikipedia at Stage 1.5 lands
  on top of a backbone that already has warm (if wobbly) weights
  for every major language Common Crawl touched.

  Multilingual fine-tuning is a separate future stage (corpus
  `D:\grillcheese_training_data\unified\allenai_c4_multilingual.500m_tokens.jsonl`
  already in hand for that run). For now, the leakage stays in
  bloopers territory — the subword merges produce mid-word
  script switches that read beautifully as accidental concrete
  poetry.
