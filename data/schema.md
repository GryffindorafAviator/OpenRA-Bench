# Results CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `agent_name` | str | Agent identifier displayed on leaderboard |
| `agent_type` | str | Category: "Scripted", "LLM", or "RL" |
| `opponent` | str | AI difficulty: "Easy", "Normal", or "Hard" |
| `games` | int | Number of games played (minimum 10) |
| `win_rate` | float | Win percentage (0.0 - 100.0) |
| `score` | float | Composite benchmark score (0.0 - 100.0) |
| `avg_kills` | float | Average enemy cost destroyed per game |
| `avg_deaths` | float | Average own cost lost per game |
| `kd_ratio` | float | Average kills_cost / deaths_cost ratio |
| `avg_economy` | float | Average final assets_value per game |
| `avg_game_length` | int | Average game duration in ticks |
| `timestamp` | str | Evaluation date (ISO 8601, YYYY-MM-DD) |
| `replay_url` | str | URL to replay file(s), empty if none |
