# Competition Writeup — March Machine Learning Mania

## Background

I discovered Kaggle about two years ago and ended up replacing my chess addiction with a Kaggle one (went from Elo maxxing to Kaggle maxxing). The community is amazing; in this timeframe, I've managed to learn and ingest more information than in my entire life before, just by getting my hands dirty in various competitions. I've also played basketball since I was 8 and even played a couple of years semi-professionally. I am a huge basketball fan, almost never miss watching a game (local league + Euroleague mostly), but this was my first year following the NCAA closely. The format is significantly different (2 halves instead of 4 quarters, for example), and the age, eligibility, and continuity (now also NIL!) factors were all things that shaped my decision-making throughout the contest. So this competition was the perfect match for me to combine my major interests: basketball and ML!

### Resources Used Outside of Kaggle

- *Basketball Analytics: Objective and Efficient Strategies for Understanding How Teams Win* — Stephen M. Shea and Christopher E. Baker
- *Basketball on Paper: Rules and Tools for Performance Analysis* — Dean Oliver
- *Mathletics: How Gamblers, Managers, and Sports Enthusiasts Use Mathematics in Baseball, Basketball, and Football* — Wayne L. Winston
- A bunch of MIT Sloan Sports Analytics Conference videos from their YouTube channel

Alas, despite the heavy preparation (more data collection, planning, new features, watching most of the games, and keeping up to date with expert analysis), life happened. I broke my finger playing basketball (the irony! — lost a month of work) and ended up getting ambushed by the competition deadline (it wasn't the usual 1 am 😄). So, with time running out due to the injury recovery and getting caught off guard by the deadline, I couldn't fully implement my primary approach nor use it effectively in two submissions. Instead, I had to use one of my submissions for a gambling approach (heavily informed by my research and watching the games, of course) which, ironically, ended up performing way better!

---

## Why Florida?

It was definitely the most aggressive call I could make, banking on a team that wasn't necessarily the consensus favorite pre-tournament, but there were several factors that made them seem like a potentially undervalued, high-upside play.

### Analytics-Driven Program

A huge part of it was their deep commitment to analytics. It really felt like the foundation of their program under Coach Golden. They had a dedicated analytics director/coach, Jonathan Safir (I am a big fan!), and it influenced pretty much everything — from player evaluation and recruiting right down to in-game strategy. You could see the tangible results too. Their win totals steadily climbed each season Golden was there.

This data-first approach wasn't just theoretical; you could see it in their decisions on the court. The examples that jump out for me: keeping a key player like Will Richard in the game against Texas Tech despite him getting three fouls in the first half. The intentional foul late in a half against Oklahoma to potentially gain an extra possession — showed they were willing to squeeze probabilities for potential, even if minor, gains (Safir's touch for sure, he worked on this with KenPom: https://kenpom.com/blog/the-guide-to-fouling-when-leading-or-tied/).

### Physical Preparation and Peaking at the Right Time

Their physical preparation also seemed advanced. They looked notably strong late in games during their SEC tournament championship run (one of the toughest, if not the toughest division), where they won three games in three days. Smart workload management obviously throughout. They were peaking at the right time.

### Roster Construction and Continuity

Then there was the way the roster was built. They didn't rely on stacking top high school recruits. In fact, that year's team was unique for Florida because it didn't feature any Top 100 players from the high school rankings! Instead, they heavily used the transfer portal and their analytics to identify players potentially overlooked by others, focusing on specific metrics beyond basic scoring stats, like defensive contributions. Key players such as Walter Clayton Jr. and Alijah Martin were actually zero-star recruits initially, later brought in from other programs. Will Richard was another example, transferring from Belmont. They seemed focused on finding specific fits and value, reportedly without engaging in major NIL bidding wars.

This unconventional roster construction was significant because it suggested Florida might have been underrated by traditional evaluation methods. But importantly, while they lacked top-ranked freshmen, they excelled in another key area: **roster continuity** (this was one of the new features introduced in my other solution, only to get destroyed by the gambling one haha). There's a clear trend in the portal era where national champions rely heavily on returning players:

| Team | Year | % Minutes from Returners |
|------|------|--------------------------|
| Kansas | 2022 | 81% |
| UConn | 2023 | 53% |
| UConn | 2024 | 61% |
| Houston | 2025 | 82% |
| Auburn | 2025 | 69% |
| **Florida** | **2025** | **70%** |
| Duke | 2025 | 22% |

Florida's 70% continuity was significantly higher than Duke's 22% among the Final Four teams. This continuity factor felt crucial; getting key players to return builds experience and cohesion.

### The Logic

Combining that experienced core with their sophisticated preparation and analytics-driven coaching painted a picture of a team uniquely equipped for a deep tournament run. It was still a gamble, of course, but one that felt supported by these specific factors. Factoring in that experience and continuity was the key to my entire logic. With Duke and Houston on the other side, I anticipated a potential final against a very talented but less experienced Duke squad (that was my bracket pick to emerge from the other side… RIP), and I felt Florida's veteran presence gave them the edge in that hypothetical championship matchup — in hindsight, I should've made the SAME call for Houston vs. Duke, for the same reasons!

---

## Final Thoughts

Looking back, remembering all the games I stayed up watching until 6 am, this Florida pick truly felt like a wild ride. It was a crazy gamble on paper (although SOME experts did agree with me, like: [32 Analytics](https://x.com/32_Analytics/status/1902730569437315463)), and watching it unfold was something else. They consistently pulled themselves out of seemingly impossible situations. I swear the ESPN win probability graphs during those games matched my heart rate spike for spike! The comebacks against Auburn (not to forget the heart attack against UConn early on…), Texas Tech, and finally Houston in the championship game — those were moments where almost any other team would have cracked (just look at how Duke folded TWICE, and finally in the Final Four game). It genuinely started to feel like maybe they were destined to win it all, given the resilience they showed time and time again.

To be honest, of course I am a little disappointed that the gamble completely overshadowed the months of work I put into my other submission. Adding to the irony, it also performed very well, finishing near the bottom of the gold medal zone on the final leaderboard! So, while the gamble obviously leapfrogged and completely destroyed it, I feel like the underlying preparation helped me comfortably "waste" a submission on the Gators. The second "losing" submission will definitely be coming back with vengeance next year though 😄

On that note, thinking about the competition format — while two submissions made this gamble possible, I actually find myself agreeing with the sentiment that maybe a single submission format would be better for future iterations. It forces a different kind of discipline. I also really hope the competition considers adopting the very creative and cool logistic Brier, as suggested by Ryan, raddar, and others.

Finally and most importantly, the basis for my winning gambling submission relied heavily on the excellent starter notebook provided by goto, using it as a foundation for the Florida overrides (I am definitely not the only one here haha). So, a huge thanks goes to the author of that notebook and the underlying [goto_conversion](https://github.com/gotoConversion/goto_conversion?tab=readme-ov-file) package — go give it a star and upvote their notebooks! I have followed their work extensively and their contributions to this competition over the years, including the great presentation shared here, are invaluable resources for me personally and the community.
