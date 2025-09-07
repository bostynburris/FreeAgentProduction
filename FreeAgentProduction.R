install.packages("stargazer")
library("stargazer")
library("readr")
mlb_teams <- read_csv("/Users/bostynburris/Desktop/FreeAgentProduction/mlb_teams.csv")
View(mlb_teams)
df <- mlb_teams

df$winpct <- df$wins / df$games_played
df$xbh <- df$doubles + df$triples + df$homeruns
df$batting_avg <- df$hits / df$at_bats
df$batter_obp <- (df$hits + df$walks + df$batters_hit_by_pitch) /
  (df$at_bats + df$walks + df$batters_hit_by_pitch + df$sacrifice_flies)
df$batter_singles <- df$hits - df$doubles - df$triples - df$homeruns
df$slugging_pct <- ((df$batter_singles + (df$doubles * 2) + (df$triples * 3) + (df$homeruns * 4)) / df$at_bats)
df$batter_ops <- df$slugging_pct + df$batter_obp
df$baa <- df$hits_allowed / (df$outs_pitches + df$strikeouts_by_pitchers)
View(df)

install.packages("dplyr")
library(dplyr)

install.packages("ggplot2")
install.packages("caret")
install.packages("caTools")

library(ggplot2)
library(caret)
library(caTools)

### plotting the effect of extra base hits on wins for class project
ggplot(df, aes(x = xbh, y = wins)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Extra Base Hits vs Wins", x = "Extra Base Hits", y = "Wins") +
  theme_minimal()


full_season_data <- df %>%
  filter(games_played == 162)
View(full_season_data)


sum(is.na(full_season_data$runs_scored))
sum(is.na(full_season_data$earned_runs_allowed))
sum(is.na(full_season_data$xbh))
sum(is.na(full_season_data$batting_avg))
sum(is.na(full_season_data$earned_run_average))
sum(is.na(full_season_data$fielding_percentage))
sum(is.na(full_season_data$strikeouts_by_batters))
sum(is.na(full_season_data$homeruns))
sum(is.na(full_season_data$walks))
sum(is.na(full_season_data$stolen_bases))
sum(is.na(full_season_data$baa))
sum(is.na(full_season_data$homeruns_allowed))
sum(is.na(full_season_data$batter_obp))
sum(is.na(full_season_data$slugging_pct))
sum(is.na(full_season_data$batter_ops))
sum(is.na(full_season_data$wins))

numerical_model_columns <- data.frame(
  runs_scored = full_season_data$runs_scored,
  earned_runs_allowed = full_season_data$earned_runs_allowed,
  xbh = full_season_data$xbh,
  batting_avg = full_season_data$batting_avg,
  earned_run_average = full_season_data$earned_run_average,
  fielding_percentage = full_season_data$fielding_percentage,
  strikeouts_by_batters = full_season_data$strikeouts_by_batters,
  homeruns = full_season_data$homeruns,
  walks = full_season_data$walks,
  steals = full_season_data$stolen_bases,
  opp_batting_avg = full_season_data$baa,
  homeruns_allowed = full_season_data$homeruns_allowed,
  on_base_pct = full_season_data$batter_obp,
  slugging_pct = full_season_data$slugging_pct,
  batter_ops = full_season_data$batter_ops,
  wins = full_season_data$wins
)

numerical_model_columns_clean <- na.omit(numerical_model_columns)

X_columns <- data.frame(
  numerical_model_columns_clean$runs_scored, 
  numerical_model_columns_clean$earned_runs_allowed,
  numerical_model_columns_clean$xbh,
  numerical_model_columns_clean$batting_avg,
  numerical_model_columns_clean$earned_run_average,
  numerical_model_columns_clean$fielding_percentage,
  numerical_model_columns_clean$strikeouts_by_batters,
  numerical_model_columns_clean$homeruns,
  numerical_model_columns_clean$steals,
  numerical_model_columns_clean$opp_batting_avg,
  numerical_model_columns_clean$walks,
  numerical_model_columns_clean$homeruns_allowed,
  numerical_model_columns_clean$on_base_pct,
  numerical_model_columns_clean$slugging_pct,
  numerical_model_columns_clean$batter_ops)

set.seed(42)
split <- sample.split(numerical_model_columns_clean$wins, SplitRatio = 0.8)

train_data <- subset(numerical_model_columns_clean, split == TRUE)
test_data <- subset(numerical_model_columns_clean, split == FALSE)

X_columns_scaled <- c(
  "runs_scored", "earned_runs_allowed", "xbh",
  "batting_avg", "earned_run_average",
  "fielding_percentage", "strikeouts_by_batters", 
  "homeruns", "steals", "opp_batting_avg", "walks", 
  "homeruns_allowed", "on_base_pct","slugging_pct", "batter_ops"
)

train_means <- sapply(train_data[X_columns_scaled], mean)
train_sds   <- sapply(train_data[X_columns_scaled], sd)

head(train_means)
head(train_sds)

# Scale training data
train_scaled <- train_data
train_scaled[X_columns_scaled] <- sweep(train_scaled[X_columns_scaled], 2, train_means, "-")
train_scaled[X_columns_scaled] <- sweep(train_scaled[X_columns_scaled], 2, train_sds, "/")

# Scale test data using same means/sds
test_scaled <- test_data
test_scaled[X_columns_scaled] <- sweep(test_scaled[X_columns_scaled], 2, train_means, "-")
test_scaled[X_columns_scaled] <- sweep(test_scaled[X_columns_scaled], 2, train_sds, "/")

head(train_scaled)
head(test_scaled)
View(test_scaled)

model <- lm(wins ~ runs_scored + earned_runs_allowed + xbh + 
              batting_avg + earned_run_average + 
              fielding_percentage + strikeouts_by_batters + homeruns + 
              steals + opp_batting_avg + walks + homeruns_allowed +
              on_base_pct + slugging_pct + batter_ops,
            data = train_scaled)

summary(model)
library(stargazer)
stargazer(model, type = "text")

library(lmtest)
library(sandwich)
robust_se <- sqrt(diag(vcovHC(model, type = "HC1")))
robust_se

### model2 cuts out statistically insignificant variables from first model
model2 <- lm(wins ~ runs_scored + xbh + 
               batting_avg + earned_run_average + 
               fielding_percentage + walks + 
               on_base_pct + slugging_pct,
             data = train_scaled)
summary(model2)
stargazer(model2, type = "text")

robust_se2 <- sqrt(diag(vcovHC(model2, type = "HC1")))
robust_se2

### running model 2 on the test set to determine accuracy and create a visual representation
y_pred <- predict(model2, newdata = test_scaled)

y_test <- subset(numerical_model_columns_clean$wins, split == FALSE)

mse <- mean((y_test - y_pred)^2)
r_squared <- cor(y_test, y_pred)^2

cat(mse)
cat(r_squared)

plot <- data.frame(Actual = y_test, Predicted = y_pred)

#summary(plot) shows the accuracy of the quartiles from predicted to actual model, proving high accuracy
summary(plot)
str(plot)

ggplot(plot, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Actual vs Predicted Wins", x = "Actual Wins", y = "Predicted Wins") +
  theme_minimal()

# Calculate the average of the 'runs_scored' column in data frame 'df'
average_winpct <- mean(df$winpct, na.rm = TRUE)
print(average_winpct)

# Calculate RMSE which gives us the amount of wins the model is off by on average 
rmse <- sqrt(mse)
print(rmse)

# Average wins for 162 games
average_wins <- 162 * average_winpct
print(average_wins)

# Calculate relative error as a percentage
# relative error shows only a 0.37% showing extremely high precision
relative_error <- (rmse / average_wins) * 100
print(relative_error)


#######################################################################################################################

#### End of Assigned Group Project ####

#######################################################################################################################

## iteration for 2024 Boston Red Sox

RedSox_2024 <- data.frame(at_bats = 5577,
                          hits = 1404,
                          runs_scored = 751,
                          xbh = 535,
                          total_bases = 2357,
                          batting_avg = .252,
                          earned_run_average = 4.05,
                          earned_runs = 654,
                          innings_pitched = 1452.2,
                          fielding_percentage = .981,
                          homeruns = 194,
                          walks = 493,
                          hbp = 73,
                          sf = 40,
                          on_base_pct = .319,
                          slugging_pct = .423
                          )
View(RedSox_2024)

model2_columns_scaled <- c(
  "runs_scored", "xbh",
  "batting_avg", "earned_run_average",
  "fielding_percentage", 
  "homeruns", "walks", 
  "on_base_pct","slugging_pct"
)

# Scale using the training means and sds
RedSox_2024_scaled <- RedSox_2024
RedSox_2024_scaled[model2_columns_scaled] <- sweep(RedSox_2024_scaled[model2_columns_scaled], 2, train_means[model2_columns_scaled], "-")
RedSox_2024_scaled[model2_columns_scaled] <- sweep(RedSox_2024_scaled[model2_columns_scaled], 2, train_sds[model2_columns_scaled], "/")

predicted_wins_RedSox_2024 <- predict(model2, newdata = RedSox_2024_scaled)
print(predicted_wins_RedSox_2024)

#######################################################################################################################

#### MLB free agency 2025 .xlsx iteration to download available free agent players

library(readxl)
library(baseballr)

freeagents2026 <- read_csv("/Users/bostynburris/Desktop/FreeAgentProduction/MLBFreeAgents2026.csv")
View(freeagents2026)

library(tidyr)
library(dplyr)

freeagentnames <- freeagents2026 %>%
  separate(Player, into = c("last", "first"), sep = ", ") %>%
  mutate(player_name = paste(first, last))
View(freeagentnames)

batting_2024 <- bref_daily_batter(t1 = "2024-04-01", t2 = "2024-10-01")
View(batting_2024)

free_agent_hitters <- batting_2024 %>%
  filter(Name %in% freeagentnames$player_name)
free_agent_hitters$xbh <- free_agent_hitters$X2B + free_agent_hitters$X3B + 
  free_agent_hitters$HR
free_agent_hitters$TB <- free_agent_hitters$X1B + (free_agent_hitters$X2B * 2) + 
  (free_agent_hitters$X3B * 3) + (free_agent_hitters$HR * 4)
View(free_agent_hitters)

pitching_2024 <- bref_daily_pitcher(t1 = "2024-04-01", t2 = "2024-10-01")
View(pitching_2024)

free_agent_pitchers <- pitching_2024 %>%
  filter(Name %in% freeagentnames$player_name)
View(free_agent_pitchers)


#### now using downloaded free agent stats from above, adding each player's
#### stats to the 2024 Red Sox totals to approximate how adding each player 
#### changes the amount of wins. then in descending order shows the players with
#### the most positive impact down to players with the worst impact


## HITTERS LOOP

evaluate_hitter_impact <- function(player_stats, team_base, model2, train_means, train_sds, feature_names) {
  
  adjusted <- team_base
  adjusted$runs_scored <- team_base$runs_scored + player_stats$R
  adjusted$xbh <- team_base$xbh + player_stats$xbh
  adjusted$homeruns <- team_base$homeruns + player_stats$HR
  adjusted$batting_avg <- (team_base$hits + player_stats$H) / (team_base$at_bats + player_stats$AB)
  adjusted$walks <- team_base$walks + player_stats$BB
  adjusted$on_base_pct <- (team_base$hits + team_base$walks + 
                             team_base$hbp + player_stats$H + 
                                         player_stats$BB + 
                                         player_stats$HBP) / (team_base$at_bats + 
                                                                team_base$walks + 
                                                                team_base$hbp + 
                                                                team_base$sf +
                                                                player_stats$AB +
                                                                player_stats$BB +
                                                                player_stats$HBP +
                                                                player_stats$SF) 
  adjusted$slugging_pct <- (team_base$total_bases + player_stats$TB) / (team_base$at_bats + 
                                                                                        player_stats$AB)
  adjusted$earned_run_average <- team_base$earned_run_average
  adjusted$fielding_percentage <- team_base$fielding_percentage
  
  scaled_adjusted <- sweep(adjusted[feature_names], 2, train_means[feature_names], "-")
  scaled_adjusted <- sweep(scaled_adjusted, 2, train_sds[feature_names], "/")
  
  # Predict
  adj_predicted_wins <- predict(model2, newdata = as.data.frame(scaled_adjusted))
  return(adj_predicted_wins)
}

results_hitters <- data.frame(Player = character(), Predicted_Wins = numeric())

feature_names = c("runs_scored", "xbh", "homeruns", "batting_avg", "earned_run_average",
                  "fielding_percentage", "walks", "on_base_pct", "slugging_pct")

baseline_wins <- predict(model2, newdata = as.data.frame(
  sweep(sweep(RedSox_2024[feature_names], 2, train_means[feature_names], "-"), 
        2, train_sds[feature_names], "/")
))

for (i in 1:nrow(free_agent_hitters)) {
  player_name <- free_agent_hitters$Name[i]
  player_stats <- free_agent_hitters[i, ]
  
  predicted <- evaluate_hitter_impact(
    player_stats = player_stats,
    team_base = RedSox_2024,
    model2 = model2,
    train_means = train_means,
    train_sds = train_sds,
    feature_names = c("runs_scored", "xbh", "homeruns", "batting_avg", "earned_run_average",
                      "fielding_percentage", "walks", "on_base_pct", "slugging_pct")
  )
  
  results_hitters <- rbind(results_hitters, data.frame(
    Player = player_name, 
    Predicted_Wins = predicted,
    Win_Change = predicted - baseline_wins
  ))
}

# Sort results in descending order
results_hitters <- results_hitters[order(-results_hitters$Predicted_Wins), ]
View(results_hitters)


## PITCHERS LOOP

evaluate_pitcher_impact <- function(player_stats, team_base, model2, train_means, train_sds, feature_names) {
  
  adjusted <- team_base
  adjusted$runs_scored <- team_base$runs_scored
  adjusted$xbh <- team_base$xbh
  adjusted$homeruns <- team_base$homeruns
  adjusted$batting_avg <- team_base$batting_avg
  adjusted$walks <- team_base$walks
  adjusted$on_base_pct <- team_base$on_base_pct
  adjusted$slugging_pct <- team_base$slugging_pct
  adjusted$fielding_percentage <- team_base$fielding_percentage
  adjusted$earned_run_average <- ((team_base$earned_runs + player_stats$ER) * 9) / 
    (team_base$innings_pitched + player_stats$IP)
    
  scaled_adjusted <- sweep(adjusted[feature_names], 2, train_means[feature_names], "-")
  scaled_adjusted <- sweep(scaled_adjusted, 2, train_sds[feature_names], "/")
  
  adj_predicted_wins <- predict(model2, newdata = as.data.frame(scaled_adjusted))
  return(adj_predicted_wins)
}

results_pitchers <- data.frame(Player = character(), Predicted_Wins = numeric())

baseline_wins <- predict(model2, newdata = as.data.frame(
  sweep(sweep(RedSox_2024[feature_names], 2, train_means[feature_names], "-"), 
        2, train_sds[feature_names], "/")
))

for (i in 1:nrow(free_agent_pitchers)) {
  player_name <- free_agent_pitchers$Name[i]
  player_stats <- free_agent_pitchers[i, ]
  
  predicted <- evaluate_pitcher_impact(
    player_stats = player_stats,
    team_base = RedSox_2024,
    model2 = model2,
    train_means = train_means,
    train_sds = train_sds,
    feature_names = feature_names
  )
  
  results_pitchers <- rbind(results_pitchers, data.frame(
    Player = player_name, 
    Predicted_Wins = predicted,
    Win_Change = predicted - baseline_wins
  ))
}

# Sort results in descending order
results_pitchers <- results_pitchers[order(-results_pitchers$Predicted_Wins), ]
View(results_pitchers)

#######################################################################################################################

#### Simulating effect of trading Rafael Devers to Giants using hitter impact evaluation function to see
#### change in expected wins by trading Devers for Jordan Hicks, Kyle Harrison, James Tibbs III, and Jose Bello

RedSox_2025_78games <- data.frame(at_bats = 2692,
                          hits = 668,
                          runs_scored = 367,
                          xbh = 250,
                          total_bases = 1117,
                          batting_avg = .248,
                          earned_run_average = 3.90,
                          earned_runs = 304,
                          innings_pitched = 701.1,
                          fielding_percentage = .977,
                          homeruns = 94,
                          walks = 264,
                          hbp = 36,
                          sf = 15,
                          on_base_pct = .322,
                          slugging_pct = .415
)
View(RedSox_2025_78games)

games_played_2025 <- 78
full_season_games <- 162
scaling_factor <- full_season_games / games_played_2025

RedSox_2025_projected <- RedSox_2025_78games

RedSox_2025_projected$at_bats <- round(RedSox_2025_78games$at_bats * scaling_factor)
RedSox_2025_projected$hits <- round(RedSox_2025_78games$hits * scaling_factor)
RedSox_2025_projected$runs_scored <- round(RedSox_2025_78games$runs_scored * scaling_factor)
RedSox_2025_projected$xbh <- round(RedSox_2025_78games$xbh * scaling_factor)
RedSox_2025_projected$total_bases <- round(RedSox_2025_78games$total_bases * scaling_factor)
RedSox_2025_projected$earned_runs <- round(RedSox_2025_78games$earned_runs * scaling_factor)
RedSox_2025_projected$innings_pitched <- round(RedSox_2025_78games$innings_pitched * scaling_factor, 1)
RedSox_2025_projected$homeruns <- round(RedSox_2025_78games$homeruns * scaling_factor)
RedSox_2025_projected$walks <- round(RedSox_2025_78games$walks * scaling_factor)
RedSox_2025_projected$hbp <- round(RedSox_2025_78games$hbp * scaling_factor)
RedSox_2025_projected$sf <- round(RedSox_2025_78games$sf * scaling_factor)

RedSox_2025_projected$batting_avg <- RedSox_2025_projected$hits / RedSox_2025_projected$at_bats
RedSox_2025_projected$on_base_pct <- (RedSox_2025_projected$hits + RedSox_2025_projected$walks + RedSox_2025_projected$hbp) /
  (RedSox_2025_projected$at_bats + RedSox_2025_projected$walks + RedSox_2025_projected$hbp + RedSox_2025_projected$sf)
RedSox_2025_projected$slugging_pct <- RedSox_2025_projected$total_bases / RedSox_2025_projected$at_bats
RedSox_2025_projected$earned_run_average <- (RedSox_2025_projected$earned_runs * 9) / RedSox_2025_projected$innings_pitched

View(RedSox_2025_projected)


model2_columns_scaled <- c(
  "runs_scored", "xbh",
  "batting_avg", "earned_run_average",
  "fielding_percentage", 
  "homeruns", "walks", 
  "on_base_pct","slugging_pct"
)

RedSox_2025_scaled <- RedSox_2025_projected
RedSox_2025_scaled[model2_columns_scaled] <- sweep(RedSox_2025_scaled[model2_columns_scaled], 2, train_means[model2_columns_scaled], "-")
RedSox_2025_scaled[model2_columns_scaled] <- sweep(RedSox_2025_scaled[model2_columns_scaled], 2, train_sds[model2_columns_scaled], "/")

predicted_wins_RedSox_2025 <- predict(model2, newdata = RedSox_2025_scaled)
print(predicted_wins_RedSox_2025)

#### Red Sox are currently on pace to win ~85 games with Devers in the lineup

#### next step is to take him out of the lineup for the remaining 84 games of 
####the season and Jordan Hicks career Bullpen ERA

Rafael_Devers_2025RedSox <- data.frame(at_bats = 272,
                                       hits = 74,
                                       runs_scored = 47,
                                       xbh = 33,
                                       total_bases = 137,
                                       batting_avg = .272,
                                       homeruns = 15,
                                       walks = 56,
                                       hbp = 4,
                                       sf = 2,
                                       on_base_pct = .401,
                                       slugging_pct = .504
)

average_RedSox_hitter <- data.frame(at_bats = 191,
                                    hits = 49,
                                    runs_scored = 47,
                                    xbh = 33,
                                    total_bases = 81,
                                    batting_avg = .247,
                                    homeruns = 7,
                                    walks = 16,
                                    hbp = 3,
                                    sf = 1,
                                    on_base_pct = .322,
                                    slugging_pct = .424)

RedSox_no_devers <- RedSox_2025_78games
numeric_cols <- colnames(Rafael_Devers_2025RedSox)

devers_84_hitter_projected <- data.frame(at_bats = Rafael_Devers_2025RedSox$at_bats,
                                         hits = Rafael_Devers_2025RedSox$hits,
                                         runs_scored = Rafael_Devers_2025RedSox$runs_scored,
                                         xbh = Rafael_Devers_2025RedSox$xbh,
                                         total_bases = Rafael_Devers_2025RedSox$total_bases,
                                         homeruns = Rafael_Devers_2025RedSox$homeruns,
                                         walks = Rafael_Devers_2025RedSox$walks,
                                         hbp = Rafael_Devers_2025RedSox$hbp,
                                         sf = Rafael_Devers_2025RedSox$sf
                                         )

devers_84_games <- devers_84_hitter_projected * (84 / 78)

average_RedSox_hitter_projected <- data.frame(at_bats = average_RedSox_hitter$at_bats,
                                         hits = average_RedSox_hitter$hits,
                                         runs_scored = average_RedSox_hitter$runs_scored,
                                         xbh = average_RedSox_hitter$xbh,
                                         total_bases = average_RedSox_hitter$total_bases,
                                         homeruns = average_RedSox_hitter$homeruns,
                                         walks = average_RedSox_hitter$walks,
                                         hbp = average_RedSox_hitter$hbp,
                                         sf = average_RedSox_hitter$sf
)
avg_RedSox_hitter_84_games <- average_RedSox_hitter_projected * (84 / 78)

RedSox_2025_hitter_projected <- data.frame(at_bats = RedSox_2025_projected$at_bats,
                                           hits = RedSox_2025_projected$hits,
                                           runs_scored = RedSox_2025_projected$runs_scored,
                                           xbh = RedSox_2025_projected$xbh,
                                           total_bases = RedSox_2025_projected$total_bases,
                                           homeruns = RedSox_2025_projected$homeruns,
                                           walks = RedSox_2025_projected$walks,
                                           hbp = RedSox_2025_projected$hbp,
                                           sf = RedSox_2025_projected$sf
                                           )

RedSox_no_devers <- RedSox_2025_hitter_projected - devers_84_games

RedSox_no_devers <- RedSox_no_devers + avg_RedSox_hitter_84_games

RedSox_no_devers$batting_avg <- RedSox_no_devers$hits / RedSox_no_devers$at_bats
RedSox_no_devers$on_base_pct <- (RedSox_no_devers$hits + RedSox_no_devers$walks + RedSox_no_devers$hbp) / 
  (RedSox_no_devers$at_bats + RedSox_no_devers$walks + RedSox_no_devers$hbp + RedSox_no_devers$sf)
RedSox_no_devers$slugging_pct <- RedSox_no_devers$total_bases / RedSox_no_devers$at_bats
RedSox_no_devers$earned_run_average <- RedSox_2025_projected$earned_run_average
RedSox_no_devers$innings_pitched <- RedSox_2025_projected$innings_pitched
RedSox_no_devers$earned_runs <- RedSox_2025_projected$earned_runs
RedSox_no_devers$earned_run_average <- RedSox_2025_projected$earned_run_average
RedSox_no_devers$fielding_percentage <- RedSox_2025_projected$fielding_percentage

adjusted_scaled <- sweep(RedSox_no_devers[model2_columns_scaled], 2, train_means[model2_columns_scaled], "-")
adjusted_scaled <- sweep(adjusted_scaled, 2, train_sds[model2_columns_scaled], "/")


wins_no_devers <- predict(model2, newdata = as.data.frame(adjusted_scaled))

#### predicted difference in wins by removing Rafael Devers ####

no_devers_change_wins <- wins_no_devers - predicted_wins_RedSox_2025
cat(no_devers_change_wins)

#### The Predicted decrease in wins by removing Devers from the lineup is 0.516 games.
#### This was calculated by replacing his stats through 78 games
#### with the average stats from the 11 current players with the most playtime. 
#### These players include(all through 78 games played): 
#### Carlos Narváez, Kristian Campbell, Trevor Story, Alex Bregman,Jarren Duran, 
#### Ceddanne Rafaela, Wilyer Abreu, Abraham Toro, Romy Gonzalez, Rob Refsnyder,
#### and David Hamilton

##################################################################################################

#### next step is to find out what adding Jordan Hicks(bullpen) and Kyle Harrison
#### to current roster since other two players not likely to be called up this year

jordanhicks_bullpen <- data.frame(earned_run_average = 3.73)
kyleharrison <- data.frame(earned_run_average = 4.48)

#### first step is to remove two weaker arms currently in staff that Hicks and Harrison
#### would likely fill in for

#### Tanner Houck having a rough year with 39 earned in 43.1 IP
#### Zack Kelly has 12 earned in 18.2 IP

#### Sending them down to triple A to make room for Hicks and Harrison

# Projected Red Sox totals
team_earned_runs <- 631
team_innings_pitched <- 1456.1

# Tanner Houck expected innings rest of season (to be removed)
houck_er <- 41.18
houck_ip <- 46.1

# Zack Kelly expected innings rest of season (to be removed)
kelly_er <- 12.29
kelly_ip <- 19.1

# Jordan Hicks estimated innings last 84 games (to be added)
hicks_er <- 10.61
hicks_ip <- 25.2

# Kyle Harrison estimated innings last 84 games (to be added)
harrison_er <- 10.45
harrison_ip <- 21.0

adj_er <- team_earned_runs - houck_er - kelly_er
adj_ip <- team_innings_pitched - houck_ip - kelly_ip

adj_er <- adj_er + hicks_er + harrison_er
adj_ip <- adj_ip + hicks_ip + harrison_ip

new_era_RedSox_2025_projected <- (adj_er * 9) / adj_ip

#### adjusted team projected ERA after 162 games is 3.75

adj_RedSox_no_devers <- RedSox_no_devers
adj_RedSox_no_devers$batting_avg <- RedSox_no_devers$hits / RedSox_no_devers$at_bats
adj_RedSox_no_devers$on_base_pct <- (RedSox_no_devers$hits + RedSox_no_devers$walks + RedSox_no_devers$hbp) / 
  (RedSox_no_devers$at_bats + RedSox_no_devers$walks + RedSox_no_devers$hbp + RedSox_no_devers$sf)
adj_RedSox_no_devers$slugging_pct <- RedSox_no_devers$total_bases / RedSox_no_devers$at_bats
adj_RedSox_no_devers$earned_run_average <- (RedSox_no_devers$earned_runs * 9) / RedSox_no_devers$innings_pitched
adj_RedSox_no_devers$innings_pitched <- RedSox_2025_projected$innings_pitched
adj_RedSox_no_devers$earned_runs <- RedSox_2025_projected$earned_runs
adj_RedSox_no_devers$earned_run_average <- new_era_RedSox_2025_projected
adj_RedSox_no_devers$fielding_percentage <- RedSox_2025_projected$fielding_percentage

adjusted_scaled <- sweep(adj_RedSox_no_devers[model2_columns_scaled], 2, train_means[model2_columns_scaled], "-")
adjusted_scaled <- sweep(adjusted_scaled, 2, train_sds[model2_columns_scaled], "/")

Devers_trade_new_projected <- predict(model2, newdata = as.data.frame(adjusted_scaled))
cat(Devers_trade_new_projected)
#### Devers trade increases team projected wins from 84.7 to 86.6 and adds
#### ~$250 million back to the budget to have flexibility to sign another 
#### big bat or bullpen arm ####

#### next is to use the Devers_trade_new_projected outcome to help rerun the 
#### hitter free agent model to see which avialable free agents goinginto 2026 are 
#### going to be the best fit for the team

############# need to update stats to today (7/29/2025) and predict the rest of
#############the season based on first 108 games

RedSox_2025_108games <- data.frame(at_bats = 3708,
                                  hits = 933,
                                  runs_scored = 524,
                                  xbh = 367,
                                  total_bases = 1581,
                                  batting_avg = .252,
                                  earned_run_average = 3.79,
                                  earned_runs = 407,
                                  innings_pitched = 967,
                                  fielding_percentage = .979,
                                  homeruns = 131,
                                  walks = 264,
                                  hbp = 49,
                                  sf = 24,
                                  on_base_pct = .322,
                                  slugging_pct = .426
)

games_played_2025_new <- 108
full_season_games <- 162
scaling_factor_new <- full_season_games / games_played_2025_new

RedSox_2025_projected_new <- RedSox_2025_108games

RedSox_2025_projected_new$at_bats <- round(RedSox_2025_108games$at_bats * scaling_factor_new)
RedSox_2025_projected_new$hits <- round(RedSox_2025_108games$hits * scaling_factor_new)
RedSox_2025_projected_new$runs_scored <- round(RedSox_2025_108games$runs_scored * scaling_factor_new)
RedSox_2025_projected_new$xbh <- round(RedSox_2025_108games$xbh * scaling_factor_new)
RedSox_2025_projected_new$total_bases <- round(RedSox_2025_108games$total_bases * scaling_factor_new)
RedSox_2025_projected_new$earned_runs <- round(RedSox_2025_108games$earned_runs * scaling_factor_new)
RedSox_2025_projected_new$innings_pitched <- round(RedSox_2025_108games$innings_pitched * scaling_factor_new, 1)
RedSox_2025_projected_new$homeruns <- round(RedSox_2025_108games$homeruns * scaling_factor_new)
RedSox_2025_projected_new$walks <- round(RedSox_2025_108games$walks * scaling_factor_new)
RedSox_2025_projected_new$hbp <- round(RedSox_2025_108games$hbp * scaling_factor_new)
RedSox_2025_projected_new$sf <- round(RedSox_2025_108games$sf * scaling_factor_new)

RedSox_2025_projected_new$batting_avg <- RedSox_2025_projected_new$hits / RedSox_2025_projected_new$at_bats
RedSox_2025_projected_new$on_base_pct <- (RedSox_2025_projected_new$hits + RedSox_2025_projected_new$walks + RedSox_2025_projected_new$hbp) /
  (RedSox_2025_projected_new$at_bats + RedSox_2025_projected_new$walks + RedSox_2025_projected_new$hbp + RedSox_2025_projected_new$sf)
RedSox_2025_projected_new$slugging_pct <- RedSox_2025_projected_new$total_bases / RedSox_2025_projected_new$at_bats
RedSox_2025_projected_new$earned_run_average <- (RedSox_2025_projected_new$earned_runs * 9) / RedSox_2025_projected_new$innings_pitched

View(RedSox_2025_projected_new)



RedSox_2025_new_scaled <- RedSox_2025_projected_new
RedSox_2025_new_scaled[model2_columns_scaled] <- sweep(RedSox_2025_new_scaled[model2_columns_scaled], 2, train_means[model2_columns_scaled], "-")
RedSox_2025_new_scaled[model2_columns_scaled] <- sweep(RedSox_2025_new_scaled[model2_columns_scaled], 2, train_sds[model2_columns_scaled], "/")

predicted_wins_RedSox_2025_new <- predict(model2, newdata = RedSox_2025_new_scaled)
print(predicted_wins_RedSox_2025_new)

#### expected wins for 2025 after playing 108 games is 88.65

evaluate_hitter_impact_new <- function(player_stats, team_base, model2, train_means, train_sds, feature_names) {
  
  adjusted <- team_base
  adjusted$runs_scored <- RedSox_2025_projected_new$runs_scored + player_stats$R
  adjusted$xbh <- RedSox_2025_projected_new$xbh + player_stats$xbh
  adjusted$homeruns <- RedSox_2025_projected_new$homeruns + player_stats$HR
  adjusted$batting_avg <- (RedSox_2025_projected_new$hits + player_stats$H) / (RedSox_2025_projected_new$at_bats + player_stats$AB)
  adjusted$walks <- RedSox_2025_projected_new$walks
  adjusted$on_base_pct <- (RedSox_2025_projected_new$hits + RedSox_2025_projected_new$walks + 
                             RedSox_2025_projected_new$hbp + player_stats$H + 
                             player_stats$BB + 
                             player_stats$HBP) / (RedSox_2025_projected_new$at_bats + 
                                                    RedSox_2025_projected_new$walks + 
                                                    RedSox_2025_projected_new$hbp + 
                                                    RedSox_2025_projected_new$sf +
                                                    player_stats$AB +
                                                    player_stats$BB +
                                                    player_stats$HBP +
                                                    player_stats$SF) 
  adjusted$slugging_pct <- (RedSox_2025_projected_new$total_bases + player_stats$TB) / (RedSox_2025_projected_new$at_bats + 
                                                                          player_stats$AB)
  adjusted$earned_run_average <- RedSox_2025_projected_new$earned_run_average
  adjusted$fielding_percentage <- RedSox_2025_projected_new$fielding_percentage
  
  scaled_adjusted <- sweep(adjusted[feature_names], 2, train_means[feature_names], "-")
  scaled_adjusted <- sweep(scaled_adjusted, 2, train_sds[feature_names], "/")
  
  # Predict
  adj_predicted_wins_new <- predict(model2, newdata = as.data.frame(scaled_adjusted))
  return(adj_predicted_wins_new)
}

results_hitters_new <- data.frame(Player = character(), Predicted_Wins = numeric(), Win_Change = numeric())

baseline_wins <- predict(model2, newdata = as.data.frame(
  sweep(sweep(RedSox_2025_projected_new[feature_names], 2, train_means[feature_names], "-"), 
        2, train_sds[feature_names], "/")
))

for (i in 1:nrow(free_agent_hitters)) {
  player_name <- free_agent_hitters$Name[i]
  player_stats <- free_agent_hitters[i, ]
  
  predicted <- evaluate_hitter_impact_new(
    player_stats = player_stats,
    team_base = RedSox_2025_projected_new,
    model2 = model2,
    train_means = train_means,
    train_sds = train_sds,
    feature_names = feature_names
  )
  
  results_hitters_new <- rbind(results_hitters_new, data.frame(
    Player = player_name, 
    Predicted_Wins = predicted,
    Win_Change = predicted - baseline_wins
  ))
}

# Sort results in descending order
results_hitters_new <- results_hitters_new[order(-results_hitters_new$Predicted_Wins), ]
View(results_hitters_new)

