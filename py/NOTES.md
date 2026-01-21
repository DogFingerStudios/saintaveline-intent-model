## Expanding intent model to handle waypoints, characters, and items

> Ok, so the commands (or intent) currently has `go_to`, `hold_position` and `follow`. It will have more commands. Also, I can see three things, similar to waypoints, what future commands will have to extract. They are: 
>
> - waypoints (we've already talked about those)
> - characters (e.g. "follow father" or "do what mother does" (where "father" and "mother" are the "characters")
> - items (e.g. "drop the grenade" or "throw the rock" (where "grenade" and "rock" are the "items")
>
> Considering this, what things should I keep in mind?

<hr/>

You’re building the “right” architecture: **intent + modifiers + target resolution**. With waypoints/characters/items all behaving like “pick one from a dynamic list,” the big things to keep in mind are about **separation of concerns, disambiguation, and failure modes**.

## 1) Keep intent words and target words in different systems

Your intent model should stay focused on **how to act**, not **what to act on**.

* **Intent BoW vocab**: verbs + command phrases + adverbs (“go”, “follow”, “hold”, “drop”, “throw”, “quietly”, “now”)
* **Target resolution** (MiniLM embeddings): nouns/entities (“5th floor bathroom”, “Dad”, “grenade”, “rock”)

If your BoW vocab starts filling up with item names and waypoint names, it will get noisier and harder to maintain.

## 2) Model outputs “target type(s)”, not the target itself

As you add commands, you’ll want the model to tell you which *kinds* of targets are required.

Example schema:

```json
{
  "intent": "throw",
  "args": {
    "item": { ... },
    "target": { "type": "character|waypoint|none", ... }
  },
  "modifiers": { "speed": "fast", "stealth": true }
}
```

Even if you keep separate heads, conceptually think “intent determines required slots”.

## 3) Make a single “EntityMatcher” that works for all lists

Waypoints, characters, and items are the same operation:

> embed input text → score against cached embeddings → pick best / topK / clarify

So write it once as a reusable class:

* `EntityMatcher<WaypointId>`
* `EntityMatcher<CharacterId>`
* `EntityMatcher<ItemId>`

Then you can tune thresholds consistently.

## 4) Expect ambiguity and build a *clarification loop*

This is the big one for games.

Ambiguity examples:

* “go to the stairs” when you have “stairs (east)” and “stairs (west)”
* “follow father” when you have “Dad” and “Father Dupont”
* “throw the rock” when you have “rock” and “sharp rock”

You need a standard “ask the player” mechanism:

* return `needsClarification=true`
* return `choices` = top 2–5 candidates
* let UI handle it quickly (single click / quick-select)

This will save you from frustration and NPC permadeath moments caused by misinterpretation.

## 5) Cache aggressively, update incrementally

Entity embedding is the expensive step, so:

* Cache embeddings for all waypoints/characters/items
* Update only when:

  * created
  * renamed
  * removed
* Store embeddings keyed by stable ID, not the display name

For items, this is especially nice because the list can be “inventory items currently held” (small and dynamic).

## 6) Use intent to gate which matchers run

Don’t match everything every time.

Examples:

* `go_to` → run waypoint matcher
* `follow` → run character matcher
* `drop` / `throw` → run item matcher
* `do what mother does` → character matcher, but intent might be “mimic” or “shadow”

This reduces false positives and improves performance.

## 7) Add aliases, not brittle parsing

For characters and items, aliases matter a lot:

* “Dad” / “Father” / “my father” / “him”
* “grenade” / “frag” / “the bomb” (if you support slang)

Instead of parsing pronouns deterministically, treat aliases as candidate strings:

* store multiple names per entity
* embed all names/aliases, map back to one canonical ID
* pick best alias match → return canonical ID

This stays “intelligent” while still robust.

## 8) Keep thresholds per entity type (they won’t be identical)

Waypoints are often multi-word and descriptive → higher semantic signal
Items can be short (“rock”, “key”) → more ambiguous
Characters can be pronouns/nicknames → tricky

So you’ll likely want:

* waypoint: `minScore` slightly lower, margin slightly lower (names are more unique)
* item: `minScore` higher or margin higher (short names collide)
* character: margin higher + better aliasing

## 9) Decide what to do when entities don’t exist

Examples:

* player says “throw grenade” but no grenade in inventory
* player says “go to central park” but no waypoint by that concept
* player says “follow mother” but mother is dead (permadeath)

You need a consistent “resolution outcome”:

* `resolved` (found)
* `not_found` (no good match)
* `invalid` (match exists but not available right now)
* `ambiguous` (needs clarification)

That integrates cleanly with your survival-horror consequences.

## 10) Think about multi-slot commands now

Some commands will require more than one entity:

* “give the medkit to Grandpa” → item + character
* “throw rock at soldier” → item + character
* “move to roof vent” → waypoint
* “place mine at stairs” → item + waypoint

So plan for intents to specify multiple required slots:

* `item`
* `recipient` (character)
* `destination` (waypoint)

And run multiple matchers on the same input (or run “reduced text” variants).

---

### A solid rule of thumb for your system

**Intent classifier decides the “verb and slots”.**
**Embedding matchers decide the “nouns”.**
**UI handles ambiguity.**
**Game logic enforces availability and consequences.**

If you want, I can propose a concrete JSON contract for *all* commands (go/follow/drop/throw/give/use/attack/guard), and a C# `CommandResolver` pipeline that routes to the right matchers and returns `Resolved/NeedsClarification/Invalid` consistently.
