# Dataset Description

Each season there are thousands of NCAA® basketball games played between Division I college basketball teams, culminating in March Madness®, the national championship men's and women's tournaments that run from mid-March until their championship games in early April. We have provided a large amount of historical data about college basketball games and teams, going back many years. Armed with this historical data, you can explore it and develop your own distinctive ways of predicting March Madness® game outcomes.

The data files incorporate both men's data and women's data. The files that pertain only to men's data will start with the letter prefix `M`, and the files that pertain only to women's data will start with the letter prefix `W`. Some files span both men's and women's data, such as Cities and Conferences.

The `MTeamSpellings` and `WTeamSpellings` files, which are listed in the bottom section below, may help you map external team references into our own Team ID structure.

> We extend our gratitude to Kenneth Massey for providing much of the historical data.
> Special Acknowledgment to Jeff Sonas of Sonas Consulting for his support in assembling the dataset for this competition.

---

## File Descriptions

All of the files are complete through February 4th of the current season. As we get closer to the tournament in mid-March, we will provide updates to these files to incorporate data from the remaining weeks of the current season.

---

## Data Section 1 — The Basics

This section provides everything you need to build a simple prediction model and submit predictions.

- Team ID's and Team Names
- Tournament seeds since 1984-85 season
- Final scores of all regular season, conference tournament, and NCAA® tournament games since 1984-85 season
- Season-level details including dates and region names
- Example submission files

> By convention, when we identify a particular season, we will reference the year that the season ends in, not the year that it starts in.

### `MTeams.csv` and `WTeams.csv`

These files identify the different college teams present in the dataset.

| Column | Description |
|--------|-------------|
| `TeamID` | A 4-digit id number, uniquely identifying each NCAA® men's or women's team. A school's TeamID does not change from one year to the next (e.g. Duke men's TeamID is 1181 for all seasons). Men's team IDs range from 1000–1999; women's team IDs range from 3000–3999. |
| `TeamName` | A compact spelling of the team's college name, 16 characters or fewer. |
| `FirstD1Season` | The first season in our dataset that the school was a Division-I school. *(Men's data only — not in WTeams.csv)* |
| `LastD1Season` | The last season in our dataset that the school was a Division-I school. Currently Division-I teams are listed with `LastD1Season=2026`. *(Men's data only — not in WTeams.csv)* |

### `MSeasons.csv` and `WSeasons.csv`

These files identify the different seasons included in the historical data, along with certain season-level properties.

| Column | Description |
|--------|-------------|
| `Season` | Indicates the year in which the tournament was played. |
| `DayZero` | The date corresponding to `DayNum=0` during that season. All game dates are aligned so that the Monday championship game of the men's tournament is on `DayNum=154`. Key reference points: national semifinals on DayNum=152, play-in games on DayNum=134-135, Selection Sunday on DayNum=132. |
| `RegionW`, `RegionX`, `RegionY`, `RegionZ` | The four tournament regions are assigned letters W, X, Y, Z. The region whose name comes first alphabetically is Region W; the region that plays Region W in the national semifinals is Region X. Of the remaining two, the one that comes first alphabetically is Region Y, and the other is Region Z. |

### `MNCAATourneySeeds.csv` and `WNCAATourneySeeds.csv`

These files identify the seeds for all teams in each NCAA® tournament. There are between 64–68 rows per year depending on play-in games. In recent years the structure has settled at 68 total teams, with four play-in games leading to the final field of 64 teams entering Round 1 (DayNum=136/137). Seeds will not be known until Selection Sunday on March 15, 2026 (DayNum=132).

| Column | Description |
|--------|-------------|
| `Season` | The year that the tournament was played in. |
| `Seed` | A 3- or 4-character identifier. The first character is W, X, Y, or Z (the region); the next two digits (01–16) are the seed within the region. Play-in teams have a fourth character (`a` or `b`) assigned based on which Team ID is lower numerically. |
| `TeamID` | The id number of the team, as specified in `MTeams.csv` or `WTeams.csv`. |

### `MRegularSeasonCompactResults.csv` and `WRegularSeasonCompactResults.csv`

Game-by-game results for historical seasons, starting with 1985 for men and 1998 for women. Covers all games from DayNum 0 through 132 (Selection Sunday).

| Column | Description |
|--------|-------------|
| `Season` | The year of the associated entry in the Seasons file. |
| `DayNum` | Integer from 0 to 132 indicating what day the game was played on. |
| `WTeamID` | ID of the winning team (W = "winning", not "women's"). |
| `WScore` | Points scored by the winning team. |
| `LTeamID` | ID of the losing team. |
| `LScore` | Points scored by the losing team. |
| `WLoc` | Location of the winning team: `H` (home), `A` (away/visiting), or `N` (neutral court). |
| `NumOT` | Number of overtime periods (integer 0 or higher). |

### `MNCAATourneyCompactResults.csv` and `WNCAATourneyCompactResults.csv`

Game-by-game NCAA® tournament results for all historical seasons. Formatted exactly like the Regular Season Compact Results. All men's games are neutral site (`WLoc=N`); some women's games may not be.

The men's tournament schedule by round (general — 2021 was slightly different; women's scheduling has varied more):

| DayNum | Round |
|--------|-------|
| 134–135 (Tue/Wed) | Play-in games (64 → 64 teams) |
| 136–137 (Thu/Fri) | Round 1 (64 → 32 teams) |
| 138–139 (Sat/Sun) | Round 2 (32 → 16 teams) |
| 143–144 (Thu/Fri) | Round 3 — Sweet Sixteen (16 → 8 teams) |
| 145–146 (Sat/Sun) | Round 4 — Elite Eight / Regional Finals (8 → 4 teams) |
| 152 (Sat) | Round 5 — Final Four / National Semifinals (4 → 2 teams) |
| 154 (Mon) | Round 6 — National Championship (2 → 1 champion) |

### `SampleSubmissionStage1.csv` and `SampleSubmissionStage2.csv`

These files illustrate the submission file format with a 50% winning percentage predicted for each possible matchup. Stage1 covers seasons 2022–2025 (for model development); Stage2 covers the current season (for actual tournament submission).

| Column | Description |
|--------|-------------|
| `ID` | A 14-character string in the format `SSSS_XXXX_YYYY`, where `SSSS` is the season, `XXXX` is the lower TeamID, and `YYYY` is the higher TeamID. |
| `Pred` | Predicted winning probability for the first (lower-ID) team identified in the ID field. |

---

## Data Section 2 — Team Box Scores

Game-by-game stats at a team level (free throws attempted, defensive rebounds, turnovers, etc.) for all regular season, conference tournament, and NCAA® tournament games since the 2003 season (men) or 2010 season (women).

Detailed Results files share the first eight columns with Compact Results files (`Season`, `DayNum`, `WTeamID`, `WScore`, `LTeamID`, `LScore`, `WLoc`, `NumOT`), plus the following additional columns for both the winning (W) and losing (L) team:

| Column | Description |
|--------|-------------|
| `FGM` | Field goals made (includes both 2-point and 3-point field goals) |
| `FGA` | Field goals attempted |
| `FGM3` | Three-point field goals made |
| `FGA3` | Three-point field goals attempted |
| `FTM` | Free throws made |
| `FTA` | Free throws attempted |
| `OR` | Offensive rebounds |
| `DR` | Defensive rebounds |
| `Ast` | Assists |
| `TO` | Turnovers committed |
| `Stl` | Steals |
| `Blk` | Blocks |
| `PF` | Personal fouls committed |

> **Note:** `FGM` is the total field goals made (2-point + 3-point). To isolate 2-point field goals, subtract `FGM3` from `FGM`. Total points = `(2 × FGM) + FGM3 + FTM`.

### `MRegularSeasonDetailedResults.csv` and `WRegularSeasonDetailedResults.csv`

Team-level box scores for regular seasons since 2003 (men) or 2010 (women). Approximately 1.5% of women's games from 2010–2012 are unavailable. All games from 2013 to present have detailed results.

### `MNCAATourneyDetailedResults.csv` and `WNCAATourneyDetailedResults.csv`

Team-level box scores for NCAA® tournaments since 2003 (men) or 2010 (women).

---

## Data Section 3 — Geography

City locations of all regular season, conference tournament, and NCAA® tournament games since the 2010 season.

### `Cities.csv`

Master list of cities that have been game locations. This file is shared between men's and women's data (no M/W prefix).

| Column | Description |
|--------|-------------|
| `CityID` | A four-digit ID number uniquely identifying a city. |
| `City` | The text name of the city. |
| `State` | The state abbreviation. Non-U.S. locations use alternative abbreviations (e.g. Cancun, Mexico = `MX`). |

### `MGameCities.csv` and `WGameCities.csv`

All games since the 2010 season with the city they were played in. Covers regular season, NCAA® tourney, and secondary tournament games. Approximately 1%–2% of women's games from 2010–2012 are not listed.

| Column | Description |
|--------|-------------|
| `Season`, `DayNum`, `WTeamID`, `LTeamID` | Four columns that uniquely identify each game. |
| `CRType` | `Regular`, `NCAA`, or `Secondary` — indicates which results file contains additional game details. |
| `CityID` | The ID of the city where the game was played, referencing `Cities.csv`. |

---

## Data Section 4 — Public Rankings

Weekly team rankings (men's teams only) for dozens of rating systems — Pomeroy, Sagarin, RPI, ESPN, etc. — since the 2003 season.

### `MMasseyOrdinals.csv`

Ordinal rankings of men's teams across many ranking methodologies, gathered by Kenneth Massey.

| Column | Description |
|--------|-------------|
| `Season` | The year of the associated entry in `MSeasons.csv`. |
| `RankingDayNum` | Ranges from 0 to 133. Represents the first day it is appropriate to use the rankings for predictions (i.e., rankings are based on game outcomes up through `DayNum - 1`). Final pre-tournament rankings have `RankingDayNum=133`. |
| `SystemName` | A (usually) 3-letter abbreviation for each distinct ranking system. |
| `TeamID` | The ID of the team being ranked. |
| `OrdinalRank` | The team's overall ranking in the system (most systems rank from #1 through #351+). |

> **Disclaimer:** Kaggle has no control over when these rankings are released. Not all systems may be available before the submission deadline.

---

## Data Section 5 — Supplements

Additional supporting information including coaches, conference affiliations, alternative team name spellings, bracket structure, and game results for NIT and other postseason tournaments.

### `MTeamCoaches.csv`

Head coach for each team in each season, with start/end DayNum ranges to capture mid-season coaching changes.

| Column | Description |
|--------|-------------|
| `Season` | The year of the associated entry in `MSeasons.csv`. |
| `TeamID` | The TeamID of the coached team. |
| `FirstDayNum`, `LastDayNum` | The continuous range of days during which the coach was the head coach. |
| `CoachName` | Coach's full name in `first_last` format (underscores for spaces, all lowercase). |

### `Conferences.csv`

Division I conferences that have existed since 1985. Shared between men's and women's data (no M/W prefix).

| Column | Description |
|--------|-------------|
| `ConfAbbrev` | Short abbreviation for the conference. |
| `Description` | Longer text name for the conference. |

### `MTeamConferences.csv` and `WTeamConferences.csv`

Conference affiliations for each team during each season, tracked historically.

| Column | Description |
|--------|-------------|
| `Season` | The year of the associated entry in the Seasons file. |
| `TeamID` | The TeamID of the team. |
| `ConfAbbrev` | The conference, as described in `Conferences.csv`. |

### `MConferenceTourneyGames.csv` and `WConferenceTourneyGames.csv`

Games that were part of post-season conference tournaments (all finishing on Selection Sunday or earlier), starting from 2001 (men) or 2002 (women).

| Column | Description |
|--------|-------------|
| `ConfAbbrev` | The conference the tournament was for. |
| `Season`, `DayNum`, `WTeamID`, `LTeamID` | Four columns that uniquely identify each game. Further details are in the Regular Season Compact/Detailed Results files. |

### `MSecondaryTourneyTeams.csv` and `WSecondaryTourneyTeams.csv`

Teams that participated in post-season tournaments other than the NCAA® Tournament (e.g. NIT, WNIT), which run in parallel with the NCAA® Tournament.

| Column | Description |
|--------|-------------|
| `Season` | The year the post-season tournament was played. |
| `SecondaryTourney` | Abbreviation of the tournament (e.g. `NIT`, `WNIT`). |
| `TeamID` | The TeamID of the participating team. |

### `MSecondaryTourneyCompactResults.csv` and `WSecondaryTourneyCompactResults.csv`

Final scores for secondary post-season tournament games. Formatted like other Compact Results files, with an added `SecondaryTourney` column. These games are played after DayNum=132 and are NOT in the Regular Season Compact Results file.

Men's tournament abbreviations: `NIT`, `CBI`, `CBC`, `CIT`, `V16` (Vegas 16), `TBC` (The Basketball Classic)
Women's tournament abbreviations: `WBI`, `WBIT`, `WNIT`

### `MTeamSpellings.csv` and `WTeamSpellings.csv`

Alternative spellings of team names, intended for associating external data with our TeamID structure.

| Column | Description |
|--------|-------------|
| `TeamNameSpelling` | Alternative spelling of the team name (always lowercase). |
| `TeamID` | The TeamID for the team with the alternative spelling. |

### `MNCAATourneySlots.csv` and `WNCAATourneySlots.csv`

Identifies how teams are paired against each other based on their seeds as the tournament progresses.

| Column | Description |
|--------|-------------|
| `Season` | The year of the associated entry in the Seasons file. |
| `Slot` | Uniquely identifies a tournament game. Play-in game slots are 3 characters (e.g. `W16`, `Z13`); regular tournament game slots are 4 characters (e.g. `R1W1`, `R2X8`), where the first two characters are the round and the last two are the expected seed of the favored team. |
| `StrongSeed` | The expected stronger-seeded team or slot (for Round 1: a team seed; for Round 2+: a prior slot). |
| `WeakSeed` | The expected weaker-seeded team or slot, assuming all favored teams have won. |

### `MNCAATourneySeedRoundSlots.csv`

Represents the men's bracket structure for any given year — for a given tournament seed, shows the bracket slot, round, and possible DayNum values for each game. *(No equivalent file for women's data due to more variable scheduling. Note: the 2021 men's tournament had unusual scheduling.)*

| Column | Description |
|--------|-------------|
| `Seed` | The tournament seed of the team. |
| `GameRound` | The round during the tournament (Round 0 = play-in games; Rounds 1/2 = first weekend; Rounds 3/4 = second weekend; Rounds 5/6 = national semifinals and finals). |
| `GameSlot` | The game slot the team would be playing in during the given GameRound. |
| `EarlyDayNum`, `LateDayNum` | The earliest and latest possible DayNums the game might be played on. |
