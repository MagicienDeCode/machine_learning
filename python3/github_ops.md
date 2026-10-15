# github

## generate ssh key and configure github account (only first time)
1. `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. go to github setting, copy **public key**. `cat .ssh/id_ed25519.pub`

## create new project
1. login to github, click Repositories, then click New
2. go to project folder in terminal
3. `git init`
4. git add remote server
5. git status
6. git add *
7. git commit -m "init commit"
8. git push