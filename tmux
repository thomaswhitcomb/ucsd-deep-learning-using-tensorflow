# Set Ctrl-J as Control Prefix
unbind C-b
bind C-j send-prefix
set -g prefix C-j

#Reload config. Prefix-r
bind r source-file ~/.tmux.conf \; display "Reloaded!"

# reset window/pane index ordering
set -g base-index 1
setw -g pane-base-index 1
#
# # scrollback
set -g history-limit 500000

# mouse support
set -g mouse off
#set -g mouse-select-pane off
#set -g mouse-resize-pane off
#set -g mouse-select-window off


setw -g mode-keys vi
bind | split-window -h
bind - split-window -v
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# Pane resizing
bind -r H resize-pane -L 5
bind -r J resize-pane -D 5
bind -r K resize-pane -U 5
bind -r L resize-pane -R 5

# color
set -g default-terminal "screen-256color"
#
# status bar
# panes
set -g pane-border-fg black
set -g pane-active-border-fg brightred

## Status bar design
# status line
set -g status-justify left
set -g status-bg default
set -g status-fg colour12
set -g status-interval 2

# messaging
set -g message-fg black
set -g message-bg yellow
set -g message-command-fg blue
set -g message-command-bg black

#window mode
setw -g mode-bg colour6
setw -g mode-fg colour0

# window status
setw -g window-status-format " #F#I:#W#F "
setw -g window-status-current-format " #F#I:#W#F "
setw -g window-status-format "#[fg=magenta]#[bg=black] #I #[bg=cyan]#[fg=colour8] #W "
setw -g window-status-current-format "#[bg=brightmagenta]#[fg=colour8] #I #[fg=colour8]#[bg=colour14] #W "
setw -g window-status-current-bg default
setw -g window-status-current-fg colour166
setw -g window-status-current-attr bright
setw -g window-status-bg green
setw -g window-status-fg black
setw -g window-status-attr reverse

# Info on left (I don't have a session display for now)
set -g status-left ''

# loud or quiet?
set-option -g visual-activity off
set-option -g visual-bell off
set-option -g visual-silence off
set-window-option -g monitor-activity off
set-option -g bell-action none

# The modes {
setw -g clock-mode-colour colour64
setw -g mode-attr bold
setw -g mode-fg colour196
setw -g mode-bg colour238

# }
# The panes {

set -g pane-border-bg colour235
set -g pane-border-fg colour235
set -g pane-active-border-bg colour236
set -g pane-active-border-fg colour240

# }
# The statusbar {

set -g status-position bottom
set -g status-bg colour235
set -g status-fg colour136
set -g status-attr default
set -g status-left ''
set -g status-right '#[fg=colour233,bg=colour241,bold] %d/%m #[fg=colour233,bg=colour245,bold] %H:%M:%S '
set -g status-right-length 50
set -g status-left-length 20

setw -g window-status-current-fg colour81
setw -g window-status-current-bg colour238
setw -g window-status-current-attr bold
setw -g window-status-current-format ' #I#[fg=colour250]:#[fg=colour255]#W#[fg=colour50]#F '

setw -g window-status-fg colour244
setw -g window-status-bg default
setw -g window-status-attr dim
setw -g window-status-format ' #I#[fg=colour237]:#[fg=colour250]#W#[fg=colour244]#F '

setw -g window-status-bell-attr bold
setw -g window-status-bell-fg colour235
setw -g window-status-bell-bg colour160

# pane number display
set-option -g display-panes-active-colour colour33 #blue
set-option -g display-panes-colour colour166 #orange

# }
# The messages {

set -g message-attr bold
set -g message-fg colour166
set -g message-bg colour235

# }
