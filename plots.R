library(ggplot2)
df = read.csv("~/dev/evolved_cartpole/random_choice_agent.tsv", 
              h=F, sep="\t", col.names=c('trial', 'total_reward'))
ggplot(df, aes(trial, total_reward)) + 
  geom_point() + 
  ggtitle('random choice agent') +
  ylim(0, 2000)

# --------------------------------------

library(ggplot2)
df = read.csv("~/dev/evolved_cartpole/random_neural_agents.tsv", 
              h=F, sep="\t", col.names=c('total_reward'))
df$trial = 1:nrow(df)
ggplot(df, aes(trial, total_reward)) + 
  geom_point() + 
  ggtitle('random neural agent') +
  ylim(0, 2000)

# --------------------------------------

library(ggplot2)

df_l = read.csv("~/dev/evolved_cartpole/cms_es_fitnesses.tsv", 
                h=T, sep="\t")
df_l$trial = 1:nrow(df_l)
df_l$run='large'

df_s = read.csv("~/dev/evolved_cartpole/small_network_test.tsv", 
              h=T, sep="\t")
df_s$trial = 1:nrow(df_s)
df_s$run='small'

df = rbind(df_s, df_l)
df$run = as.factor(df$run)
df$fitness = df$fitness * -1

ggplot(df, aes(trial, fitness)) + 
  geom_point(aes(colour=run)) + 
  ggtitle('cma es agent') +
  geom_smooth(aes(colour=run)) +
  ylim(0, 2000)  
