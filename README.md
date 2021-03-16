# Capstone

## Introduction:

As the NBA season just reached it's halfway point rolling past the All-Star Break, the focus now shifts to the trading deadline. The trade deadline is a slated for March 25th this year and marks the final date that teams are allowed to initiate trades to their current season roster. Trades allow teams to move in many directions, either acquiring new talent in hopes of making a deeper run in the playoffs or sometimes swapping players for draft picks with hopes of obtaining future stars in the making via the NBA draft. Whatever a team's situation may be, there is more often than not, buzz circulating this time of year.

The case is no different for my favorite team, The Boston Celtics. The Celtics entered the year with quite a bit of optimism with the 4 stars: Jayson Tatum, Jaylen Brown, Kemba Walker, and defensive specialist Marcus Smart. While the Celtics enjoyed success early in the season, the previous 20 games have been less than kind and the Celtics find themselves sitting at a modest 19-17 record, good enough for the 4th spot in the Eastern Conference. While 8 teams make the playoffs per conference, the Celtics are amongst a dogpile of teams bottlenecked in the heap of possible playoff contenders. Currently there's only 2 games separating the Celtics from their current 4th seed and 3 other teams outside the 8th seed looking in. With so much contention for a playoff spot, the Celtics have approached me to help provide analysis to their current roster. They have contacted me with hopes of establishing whether or not each player on their roster is performing at a "playoff caliber" level or if not if it would be wise to try and trade them.

### Problem Statement:

Using both supervised and unsupervised machine learning, can I create a classification model to determine whether a current player and their statistics thus far this season is "playoff caliber"?

### Problem Solving Method:

For tackling this problem, I will plan to use some form of classification model based on the previous 30 years of player data. In order to answer this appropriately, first I need to establish what "playoff caliber" qualifies as. The most sensible approach will be to just analyze a player's performance that year and observe if they made the playoffs for that year. Assuming the relationship between a players performance and the team making the playoffs exists, then we should be able to create a model which will classify if that player's statistics are "good enough" to make the playoffs. Also since we want to compare players appropriately and accurately we will need to make tiers/classes of players. Intuitively star players and franchise players should perform at different levels compared to a more support player who may not play as much.  

### Data Sources:

Data for each player was scraped from https://www.basketball-reference.com/. The data I elected to use was just the cut and dry basic stats points per game, rebounds, blocks, steals, assists, but I also wanted to incorporate advanced metrics so called upong Player Efficiency Rating, True Shooting Percentage, Box Score Plus Minus, and Win Shares per 48 Mins. An expanded explanation for each metric can be found in the below data dictionary:

|Feature|Type|Dataset|Description|
|---|---|---|---|
|Name|object|all|Name of corresponding player|
|Team|object|all|Abbreviation of the corresponding team for each player|
|Year|int|all|the corresponding season year for each row|
|games_played|int|df_basic|the total number of games played by the player for that season|
|mpg|float|df_basic|minutes per game, number of minutes played each game by the player|
|fg%|float|df_basic|the overall average fg% for the player across that season|
|3pfg%|float|df_basic|the overall average 3pt fg% for the player across that season|
|rebounds|float|df_basic|average number of rebounds by player per game|
|assists|float|df_basic|average number of assists by player per game|
|steals|float|df_basic|average number of steals by player per game|
|blocks|float|df_basic|average number of blocks by player per game|
|points_per_game|float|df_basic|average number of points scored by player per game|
|position|object|df_basic|the position that the player plays|
|player_efficiency|float|df_advanced|The player efficiency rating (PER) is a rating of a player's per-minute productivity|
|true_shooting|float|df_advanced|True shooting percentage is a measure of shooting efficiency that takes into account field goals, 3-point field goals, and free throws|
|plus_minus|float|df_advanced|BPM uses a player’s box score information, position, and the team’s overall performance to estimate the player’s contribution in points above league average per 100 possessions played|
|win_shares_percentage|float|df_advanced|The statistic takes into account the various things a basketball player does to win or lose a game, and weighs them appropriately to provide a Win Share|
|in_playoff|int|df_advanced|if the player's team made the playoffs for that given statistical year|
|is_playoff_caliber|int|final|whether or not a player is deemed playoff caliber by the classification model created|

Here's a bit more info on some of the advanced metrics and what they mean:
Player Efficiency Rating: https://www.espn.com/nba/columns/story?columnist=hollinger_john&id=2850240
True Shooting Percentage: https://www.basketball-reference.com/about/glossary.html
Box Plus Minus: https://www.basketball-reference.com/about/bpm2.html
Win Shares Per 48: https://www.sportingcharts.com/dictionary/nba/win-shares-per-48-minutes.aspx#:~:text=The%20statistic%20takes%20into%20account,of%20wins%20for%20the%20team

### Cleaning/EDA:

Cleaning was fairly easy as every null value could just be replaced with a 0. So for any true center that never attempted a 3 pointer in their life, we can just replace that null value with a 0%. One challenge was merging the dataframes together accurately to avoid joining statistical rows multiple times for players (which is almost all) who played multiple seasons. My remedy for this was to simply perform an inner join on both the players name along with their team and the associated year for that row. 

EDA consisted of the usual suspects of examining distributions, heatmaps, and scatterplots. As is the hope with most EDA processes, I wanted to see if I could discover any insights that would lead me to believe that I could solve the problem statement with the given data or if I had to seek an alternative approach. 

I also wanted to use KMeans modeling during EDA just to observe any differences in our data across the 3 decades. There's always talk about how much the game has changed, so if that were the case we would see that in our data. As noted, this could be done by just ceating new dataframes by decades, but KMeans clustering allowed me to quickly create these clusters and succinctly show any statistical trends observed across the 30 year span. 

**Overall EDA Findings:** Based on the EDA and KMeans modeling, I feel confident we should be able to help answer the problem statement through machine learning. We should be able to classify whether a player is performing at a playoff caliber level based on the previous 30 years of player data. By establishing tiers and contextualizing our players comparitively, we should be able to identify players who are poised to either lead or contribute to their teams in ways that will result in a playoff bid.

### Modeling:

After trying a handful of classification models, the most effective models that yielded the best accuracy scores with my features were LogisticRegression and Support Vector Machine (SVM). I wanted to ensure that I tested each model on each tier as results will most likely vary, so my total analysis included a combination of LR and SVM models. I was able to improve on the baseline scores across all four tiers, with the greatest improvement coming from the more sought after "major players" tier. As we gleamed from our EDA, we would most likely see a performance drop off as we went down tiers, which was highlighted by our decreasing degrees of baseline improvement with each tier. 

For Modeling, I would like to make a few observations very clear. Firstly, that this model is only predicting based on a player's current performance, not their full year anticipated stats. The idea is to measure how they have performed thus far and if their team is in proper position from a roster standpoint to make a playoff push. Ideally we would like to know how they will fare over a whole season, but that would require different degrees of modeling and would be much harder to predict at this time. So while the scores were favorable, the model is still imperfect (as with most models). Despite the imperfections we were still able to acquire valuable insight and determine the composition of a team from a "playoff caliber" standpoint.

I would also like to continue my modeling and try to create more tiers that will make player comparison more sensible. While it still makes sense to compare players based on their minutes per game, as those high mpg players are expected to be your stars and franchise players regardless of position, I think it would be interesting to explore how the modeling accuracy changes when we incorporate positions into them. So we build a model for "Major Shooting Guards" and for "Major Centers" and so on...

### Recommendations:

The full details of model analysis and recommendations can be found in Notebook-03 as well.

**Major Player Analysis and Recommendations:** 

On the surface, not a very good result. Of the Celtics Power Four players, only one, Marcus Smart is viewed as a playoff caliber player at the halfway mark. For any avid Celtics fan, this may not come as the biggest surprise as the Celtics have experienced a myriad of woes in the early goings ranging from untimely injuries to just disappointing close losses. While all four of these players have had terrific seasons in the past and certainly possess a lot of potential it is no secret that the Celtics season to date has been a disappointing one. Fairly recently their GM and Director of Operations, Danny Ainge, had this to say about his ballclub:

https://985thesportshub.com/2021/02/18/danny-ainge-says-celtics-do-not-have-a-championship-roster/

As we've highlighted throughout the report, these less than desirable records and their blame typically fall at the feet of your stars (major players). 

This may go against the intuition of my own model, but I don't suggest seeking a trade or a change from any of these current players. While they may have underperformed over the course of the first half of the season, I believe there are enough underlying factors to contradict what the model is suggesting. In the case of Kemba Walker, he is coming off a knee surgery so has been struggling to find his rhythm, but his play has been trending upwards and returning to his former playoff caliber self. But again, it's not surprise to see that the Celtics GM reportedly was shopping Kemba around this season:

https://www.masslive.com/celtics/2021/02/kemba-walker-trade-rumors-celtics-tried-like-hell-to-trade-guard-danny-ainge-knew-knee-wasnt-right-report.html

https://www.espn.com/nba/story/_/id/30984749/why-jayson-tatum-jaylen-brown-enough-boston-celtics-season

Jayson Tatum and Jaylen Brown both nothced their 2nd and 1st All Star appearances (respectively) so despite a lackluster 1st half, possess enough potential and talent to turn it around. These two are newly annointed franchise players at the absurdly young age of 23 & 24 years old and along with their recently inked large contracts would be reason enough to continue to give them a chance. It may not be this season, but hopefully the scouting pays off and they establish themselves well past "playoff caliber."

https://www.celticsblog.com/2021/3/11/22321297/the-evolution-of-jayson-tatums-shot-making-ability-boston-celtics-nba

**Core Player Analysis and Recommendations:** 

Again, not what you would want to see from you main core players. While it is nice to see Daniel Theis earning his playoff caliber stripes, who has been playing quite well of late, you would like to see more production of players like Tristan Thompson. Albeit he may impact the game from a non-box score standpoint, does pose as an interesting potential to trade and provide upgrades to the PF position. 

Payton Pritchard, a rookie, has been a bright spot for the Celtics this year provided often needed hustle and timely shooting. While he doesn't notch a playoff caliber classification, he is shooting a healthy 40% from beyound the arc, which if we recall from our clustering via EDA, 3 point shooting has become a major focal point of the modern game. His shooting paired with simply being a rookie, I recommend keeping him but would certainly field any offers from teams if the Celtics are looking to make a much needed upgrade to their core group. 

Theis, the only playoff caliber of the bunch might make the most sense if the Celtics are looking to make a trade. Theis is a Center who can stretch the floor and shoot a high percentage from beyond the arc, the only question is his trade value. For a team looking to inject some shooting paired with size, Daniel and the Celtics could make a great trading partner.

**Rotational Player Analysis and Recommendations:**

Continuing the disappointing trend, the Celtics rotational players trend in a non playoff caliber direction. But a bright to spot to note is that as I have rerun my model from earlier this week until now (with updated stats), Robert Williams went from non playoff caliber to playoff caliber. This would make sense for anybody who has been watching him recently and can see how much of an impact he has been making to the roster. He's playing efficiently and at a higher clip after earning Coach Steven's trust.

Improvements can clearly be made in the depth department, so I would suggest exploring trades for any of these players with the exception of Robert Williams (despite being non caliber through my model). Again just another highlight of how it's important to leverage the model instead of solely relying on it to make our decisions. 

**Reserve Player Analysis and Recommendations:**

It's nice to see fan favorite Tacko Fall make his way as a playoff caliber reserve player. Tacko doesn't play all to often, but at his size 7'5" he is always a treat to play. It also seems that when he does get an opportunity he is rather efficient, generating a 22.4 player efficiency score. Although, he is usually playing against fellow reserves, so be mindful to not get overly excited.

Tacko Fall is a fan icon! 