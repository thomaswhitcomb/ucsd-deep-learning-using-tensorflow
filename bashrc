export CLICOLOR=1
#export LSCOLORS=exfxcxdxbxegedabagacad
export LSCOLORS=GxFxCxDxBxegedabagaced

parse_git_branch() {
  git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}

set -o vi
alias rm='rm -i'
alias mv='mv -i'
alias cp='cp -i'
alias ll='ls -l'
alias la='ls -a'
alias h='history'
alias lsd='ls -l | grep ^d'
alias l.='ls -a | grep "^\."'
alias tmux="TERM=screen-256color-bce tmux"
export AWS_DEFAULT_REGION=us-west-2
export PS1="DOCKER \W\[\033[32m\]\$(parse_git_branch)\[\033[00m\] $ "
source ~/git-completion.bash
alias rm='rm -i'
alias clone='git clone https://github.com/thomaswhitcomb/ucsd-deep-learning-using-tensorflow.git'
alias graph='tensorboard --port 6066 --logdir '

