from gymnasium.envs.registration import register

register(
    id="VSS-v0", entry_point="rsoccer_gym.vss.env_vss:VSSEnv", max_episode_steps=1200
)

register(
    id="SSLStaticDefenders-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge.static_defenders:SSLHWStaticDefendersEnv",
    kwargs={"field_type": 2},
    max_episode_steps=1000,
)

register(
    id="SSLDribbling-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge.dribbling:SSLHWDribblingEnv",
    max_episode_steps=4800,
)

register(
    id="SSLContestedPossession-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge.contested_possession:SSLContestedPossessionEnv",
    max_episode_steps=1200,
)

register(
    id="SSLPassEndurance-v0",
    entry_point="rsoccer_gym.ssl.ssl_hw_challenge:SSLPassEnduranceEnv",
    max_episode_steps=1200,
)

register(
    id="VSS-tcc_progressivo-v0",
    entry_point="rsoccer_gym.vss.env_vss:vss_tcc_progressivo",
    max_episode_steps=1000,
)

register(
    id="VSS-tcc_progressive-with-goalkeeper-v0",
    entry_point="rsoccer_gym.vss.env_vss:vss_tcc_progressive_with_goalkeeper",
    max_episode_steps=1000,
)

register(
    id="VSS-progressive_simple_goalkeeper-v0",
    entry_point="rsoccer_gym.vss.env_vss:vss_progressive_simple_goalkeeper",
    max_episode_steps=1000,
)

register(
    id="VSS-dumb_attacker-v0",
    entry_point="rsoccer_gym.vss.env_vss:dumb_attacker",
    max_episode_steps=300,
)

register(
    id="VSS-goleiro-v0",
    entry_point="rsoccer_gym.vss.env_vss:vss_goleiro",
    max_episode_steps=1000,
)

register(
    id="VSS-progressive-atk-random-gk-v0",
    entry_point="rsoccer_gym.vss.env_vss_progressive_attacker_with_goalkeeper:VSSProgressiveAttackerVSGoalkeeper",
    max_episode_steps=1000,
)

register(
    id="VSS-progressive-atk-half-gk-v0",
    entry_point="rsoccer_gym.vss.env_vss_progressive_attacker_with_goalkeeper:VSSProgressiveAttackerVSHalfGoalkeeper",
    max_episode_steps=1000,
)

register(
    id="VSS-progressive-atk-gk-v0",
    entry_point="rsoccer_gym.vss.env_vss_progressive_attacker_with_goalkeeper:VSSProgressiveAttackerVSGoalkeeper",
    max_episode_steps=1000,
)
