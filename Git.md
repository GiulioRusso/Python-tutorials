<center><h1>🤝 Collaborative Git Workflow Guide</h1></center>

A comprehensive guide for working on shared projects with Git, focusing on proper branching, pulling, merging, and conflict resolution.

<br>

## 📋 Table of Contents
1. [The Basic Workflow](#the-basic-workflow)
2. [Initial Setup](#initial-setup)
3. [Daily Workflow Step-by-Step](#daily-workflow-step-by-step)
4. [Understanding Branches](#understanding-branches)
5. [Handling Merge Conflicts](#handling-merge-conflicts)
6. [Best Practices](#best-practices)
7. [Common Scenarios](#common-scenarios)
8. [Troubleshooting](#troubleshooting)

<br>

## 🔄 The Basic Workflow

Here's the **golden rule** for collaborative Git work:

```
1. Create a branch for your work
2. Make your changes on that branch
3. Pull the latest changes from remote
4. Merge remote changes into YOUR branch (solve conflicts here)
5. Push your branch to remote
6. Create a Pull Request (PR) / Merge Request (MR)
7. After review, merge into main branch
```

**Why this workflow?** 
- ✅ Keeps the main branch stable and working
- ✅ Allows you to work without breaking others' code
- ✅ Conflicts happen on YOUR branch, not the main branch
- ✅ Easy to review and test changes before merging

<br>

## 🚀 Initial Setup

### Step 1: Clone the Repository (First Time Only)

```bash
# Clone the project to your local machine
git clone https://github.com/username/project-name.git

# Navigate into the project folder
cd project-name
```

### Step 2: Check Your Git Configuration

```bash
# Set your name and email (if not already set)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify your configuration
git config --list
```

### Step 3: Understand the Current State

```bash
# See what branch you're on (usually 'main' or 'master')
git branch

# See the remote repository URL
git remote -v

# Check the status of your working directory
git status
```

<br>

## 📅 Daily Workflow Step-by-Step

Let's walk through a complete work session from start to finish.

### Step 1: Start with a Clean Main Branch

Before creating a new branch, always make sure your main branch is up to date.

```bash
# Switch to the main branch
git checkout main

# Pull the latest changes from remote
git pull origin main
```

**What this does:**
- `git checkout main` → Switches you to the main branch
- `git pull origin main` → Downloads and merges the latest changes from the remote repository

<br>

### Step 2: Create a New Branch for Your Work

**Never work directly on the main branch!** Always create a feature branch.

```bash
# Create and switch to a new branch in one command
git checkout -b feature/my-new-feature

# Alternative: Create branch then switch to it
git branch feature/my-new-feature
git checkout feature/my-new-feature
```

**Branch naming conventions:**
- `feature/add-login-page` → New features
- `bugfix/fix-null-pointer` → Bug fixes
- `hotfix/security-patch` → Urgent fixes
- `docs/update-readme` → Documentation updates

<br>

### Step 3: Make Your Changes

Now you can work on your code! Edit files, add features, fix bugs, etc.

```bash
# Check what files you've changed
git status

# See the specific changes you've made
git diff
```

<br>

### Step 4: Commit Your Changes Locally

Once you've made some progress, save your work with commits.

```bash
# Stage specific files
git add filename.py

# Or stage all changed files
git add .

# Commit with a descriptive message
git commit -m "Add user authentication feature"
```

**Good commit messages:**
- ✅ "Add login form validation"
- ✅ "Fix null pointer exception in user service"
- ✅ "Update API documentation for /users endpoint"

**Bad commit messages:**
- ❌ "fixed stuff"
- ❌ "changes"
- ❌ "asdf"

<br>

### Step 5: Pull Latest Changes from Remote

**This is crucial!** Before pushing your work, get the latest changes from your teammate.

```bash
# First, switch to main and update it
git checkout main
git pull origin main

# Switch back to your branch
git checkout feature/my-new-feature
```

<br>

### Step 6: Merge Main into Your Branch

This is where you handle conflicts **on your branch**, not on main!

```bash
# Merge the latest main branch into your feature branch
git merge main
```

**Possible outcomes:**

**A) No conflicts (Automatic merge):**
```
Auto-merging file.py
Merge made by the 'recursive' strategy.
```
✅ Great! Git merged everything automatically.

**B) Merge conflicts:**
```
Auto-merging file.py
CONFLICT (content): Merge conflict in file.py
Automatic merge failed; fix conflicts and then commit the result.
```
⚠️ You need to resolve conflicts manually (see next section).

<br>

### Step 7: Resolve Conflicts (If Any)

When Git can't automatically merge, you'll see conflict markers in your files.

**Example conflict in `app.py`:**
```python
def calculate_total(price, tax):
<<<<<<< HEAD
    # Your changes
    return price * (1 + tax)
=======
    # Changes from main branch
    return price + (price * tax)
>>>>>>> main
```

**How to resolve:**

1. **Open the conflicted file** in your editor
2. **Look for conflict markers:**
   - `<<<<<<< HEAD` → Your changes start here
   - `=======` → Separator between changes
   - `>>>>>>> main` → Changes from main branch end here

3. **Decide which changes to keep:**
   - Keep yours
   - Keep theirs
   - Keep both (combine them)
   - Write something completely new

4. **Remove the conflict markers** and edit to your final version:
```python
def calculate_total(price, tax):
    # Combined version - best of both
    result = price + (price * tax)
    return result
```

5. **Mark as resolved:**
```bash
# Stage the resolved file
git add app.py

# Check status to see if all conflicts are resolved
git status

# Complete the merge with a commit
git commit -m "Merge main into feature/my-new-feature, resolved conflicts"
```

<br>

### Step 8: Push Your Branch to Remote

Now your branch is ready to share with your teammate!

```bash
# Push your branch to the remote repository
git push origin feature/my-new-feature

# If this is the first time pushing this branch, you might need:
git push -u origin feature/my-new-feature
```

**What this does:**
- Uploads your branch to GitHub/GitLab/Bitbucket
- Makes it available for your teammate to review
- The `-u` flag sets up tracking (only needed first time)

<br>

### Step 9: Create a Pull Request (PR)

**On GitHub/GitLab/Bitbucket web interface:**

1. Go to your repository's website
2. Click "Pull Requests" or "Merge Requests"
3. Click "New Pull Request"
4. Select:
   - **Base branch:** `main` (where you want to merge TO)
   - **Compare branch:** `feature/my-new-feature` (your branch)
5. Add a title and description
6. Request review from your teammate
7. Click "Create Pull Request"

**Your teammate will:**
- Review your code
- Leave comments/suggestions
- Approve or request changes

<br>

### Step 10: Address Review Feedback (If Needed)

If your teammate requests changes:

```bash
# Make the requested changes to your files

# Stage and commit the changes
git add .
git commit -m "Address review feedback: improve error handling"

# Push the new commits to your branch
git push origin feature/my-new-feature
```

The Pull Request will automatically update with your new commits!

<br>

### Step 11: Merge to Main

Once approved, merge your branch into main (usually done via the web interface):

1. Click "Merge Pull Request" on GitHub/GitLab
2. Choose merge type (usually "Merge commit" or "Squash and merge")
3. Confirm the merge
4. Delete the feature branch (optional but recommended)

**Or via command line:**
```bash
# Switch to main
git checkout main

# Pull the latest (your merged changes)
git pull origin main

# Delete your local feature branch (it's merged now)
git branch -d feature/my-new-feature

# Delete the remote feature branch
git push origin --delete feature/my-new-feature
```

<br>

## 🌳 Understanding Branches

Think of branches as **parallel universes** for your code:

```
main:           A --- B --- C --- F --- G
                       \           /
feature branch:         D --- E ---
```

- **main**: The stable, production-ready code
- **feature branch**: Your experimental/development work
- **A, B, C**: Commits on main before you branched off
- **D, E**: Your commits on the feature branch
- **F**: Changes your teammate made on main while you worked
- **G**: The merge commit bringing your work back to main

<br>

### Viewing Branches

```bash
# List all local branches (* shows current branch)
git branch

# List all branches including remote
git branch -a

# See branch history as a graph
git log --oneline --graph --all
```

<br>

### Switching Branches

```bash
# Switch to an existing branch
git checkout branch-name

# Create and switch to a new branch
git checkout -b new-branch-name

# Switch using the newer 'switch' command (Git 2.23+)
git switch branch-name
git switch -c new-branch-name
```

<br>

## ⚔️ Handling Merge Conflicts

### Types of Conflicts

**1. Content Conflict** (most common)
Both you and your teammate edited the same lines in the same file.

**2. Delete/Modify Conflict**
You modified a file that your teammate deleted (or vice versa).

**3. Rename Conflict**
Both of you renamed the same file to different names.

<br>

### Conflict Resolution Strategy

**Step-by-step process:**

1. **Don't panic!** Conflicts are normal in collaboration.

2. **Check which files have conflicts:**
```bash
git status
```
Look for files marked as "both modified"

3. **Open conflicted files** and look for conflict markers

4. **Understand both changes:**
   - Read your changes carefully
   - Read your teammate's changes
   - Understand the intent of both

5. **Decide on the resolution:**
   - Talk to your teammate if unclear
   - Test the combined code
   - Make sure functionality isn't broken

6. **Edit the file** to the correct final state

7. **Remove ALL conflict markers** (`<<<<<<<`, `=======`, `>>>>>>>`)

8. **Test your code** to ensure it works

9. **Stage the resolved files:**
```bash
git add resolved-file.py
```

10. **Complete the merge:**
```bash
git commit -m "Resolve merge conflicts"
```

<br>

### Example Conflict Resolution

**Original file (before changes):**
```python
def greet(name):
    return "Hello, " + name
```

**Your change (on feature branch):**
```python
def greet(name):
    return f"Hello, {name}!"  # Using f-string
```

**Teammate's change (on main):**
```python
def greet(name):
    return "Hi there, " + name  # Changed greeting
```

**After merge attempt:**
```python
def greet(name):
<<<<<<< HEAD
    return f"Hello, {name}!"  # Using f-string
=======
    return "Hi there, " + name  # Changed greeting
>>>>>>> main
```

**Your resolution (combining both improvements):**
```python
def greet(name):
    return f"Hi there, {name}!"  # Using f-string AND new greeting
```

<br>

### Tools for Resolving Conflicts

**Visual merge tools** make conflicts easier to handle:

```bash
# Configure a merge tool (VS Code, for example)
git config --global merge.tool vscode
git config --global mergetool.vscode.cmd 'code --wait --merge $REMOTE $LOCAL $BASE $MERGED'

# Use the merge tool
git mergetool
```

**Popular merge tools:**
- VS Code (built-in Git support)
- GitKraken (visual Git client)
- Beyond Compare
- Meld
- P4Merge
- KDiff3

<br>

## ✨ Best Practices

### 1. Commit Often, Push Regularly

```bash
# Make small, focused commits
git commit -m "Add email validation"
git commit -m "Add password strength checker"
git commit -m "Add login form styling"

# Don't wait days to push
git push origin feature/login-system
```

**Benefits:**
- Easier to track changes
- Easier to revert if something breaks
- Your work is backed up remotely

<br>

### 2. Pull Before You Push

**Always update your branch before pushing:**

```bash
# Daily routine:
git checkout main
git pull origin main
git checkout your-feature-branch
git merge main
# Resolve any conflicts
git push origin your-feature-branch
```

<br>

### 3. Use Descriptive Branch Names

```bash
# Good branch names
git checkout -b feature/user-authentication
git checkout -b bugfix/login-redirect-loop
git checkout -b docs/api-documentation

# Bad branch names
git checkout -b test
git checkout -b fixes
git checkout -b my-branch
```

<br>

### 4. Write Clear Commit Messages

**Format:**
```
<type>: <short description>

<detailed description if needed>
```

**Examples:**
```bash
git commit -m "feat: Add password reset functionality"
git commit -m "fix: Resolve null pointer in user service"
git commit -m "docs: Update README with setup instructions"
git commit -m "refactor: Simplify authentication logic"
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

<br>

### 5. Keep Branches Short-Lived

- Create a branch for a specific task
- Merge it within a few days
- Delete the branch after merging
- Don't let branches live for weeks

<br>

### 6. Communicate with Your Teammate

- **Before starting:** "I'm working on the login feature"
- **During work:** "I'm modifying the User model"
- **Before big changes:** "I need to refactor the database layer"
- **When stuck:** "I'm getting a conflict in auth.py, can we discuss?"

<br>

## 📖 Common Scenarios

### Scenario 1: You Started Working on Main by Mistake

```bash
# You're on main and made changes (but haven't committed)
git status  # Shows modified files

# Create a new branch with your changes
git checkout -b feature/my-work

# Now your changes are on the new branch!
git add .
git commit -m "Add my work"
```

<br>

### Scenario 2: You Committed to Main by Mistake

```bash
# You committed to main instead of a feature branch
git log  # See your commits

# Create a new branch from current state
git branch feature/my-work

# Reset main to remote state (this removes your commits from main)
git checkout main
git reset --hard origin/main

# Switch to your feature branch (your commits are here)
git checkout feature/my-work
```

<br>

### Scenario 3: Your Teammate Pushed While You Were Working

```bash
# You're on your feature branch and ready to push
# But first, update with their changes

git checkout main
git pull origin main

git checkout feature/your-branch
git merge main
# Resolve conflicts if any

git push origin feature/your-branch
```

<br>

### Scenario 4: You Need to Update Your PR with New Main Changes

```bash
# Your PR is open, but main has new commits
git checkout main
git pull origin main

git checkout feature/your-branch
git merge main
# Resolve conflicts

git push origin feature/your-branch
# The PR automatically updates!
```

<br>

### Scenario 5: You Want to Discard Your Local Changes

```bash
# Discard all uncommitted changes
git checkout .

# Or reset to last commit
git reset --hard HEAD

# Discard changes to a specific file
git checkout filename.py
```

<br>

### Scenario 6: You Need to See What Your Teammate Changed

```bash
# See commits on main that you don't have
git fetch origin
git log HEAD..origin/main

# See the actual changes
git diff main origin/main

# See commits in a nicer format
git log --oneline --graph origin/main
```

<br>

## 🆘 Troubleshooting

### Problem: "Your branch is ahead of 'origin/main' by X commits"

**Solution:**
```bash
# Push your commits
git push origin your-branch-name
```

<br>

### Problem: "Your branch is behind 'origin/main' by X commits"

**Solution:**
```bash
# Pull the latest changes
git pull origin main
```

<br>

### Problem: "fatal: refusing to merge unrelated histories"

**Solution:**
```bash
# Allow merging unrelated histories (rare, usually on first merge)
git pull origin main --allow-unrelated-histories
```

<br>

### Problem: "error: failed to push some refs"

**Solution:**
```bash
# Someone pushed before you, pull first
git pull origin your-branch-name
# Resolve any conflicts
git push origin your-branch-name
```

<br>

### Problem: "You have divergent branches"

**Solution:**
```bash
# Your local branch and remote branch have different commits
# Option 1: Merge remote changes into local
git pull origin your-branch-name

# Option 2: Rebase your changes on top of remote (cleaner history)
git pull --rebase origin your-branch-name
```

<br>

### Problem: "I want to undo my last commit"

**Solution:**
```bash
# Keep changes, just undo commit
git reset --soft HEAD~1

# Discard changes and commit completely
git reset --hard HEAD~1

# Already pushed? Create a new commit that reverses it
git revert HEAD
```

<br>

### Problem: "Merge conflicts are too complex"

**Solution:**
```bash
# Abort the merge and start over
git merge --abort

# Go back to state before merge
git checkout your-branch-name

# Talk to your teammate about the conflicts
# Consider pair programming the merge
```

<br>

## 🎯 Quick Reference Commands

```bash
# Setup
git clone <url>                          # Clone a repository
git config --global user.name "Name"     # Set your name
git config --global user.email "email"   # Set your email

# Daily workflow
git status                               # Check current state
git pull origin main                     # Update main branch
git checkout -b feature/name             # Create new branch
git add .                                # Stage all changes
git commit -m "message"                  # Commit changes
git push origin branch-name              # Push to remote

# Branching
git branch                               # List local branches
git branch -a                            # List all branches
git checkout branch-name                 # Switch branches
git branch -d branch-name                # Delete local branch

# Merging
git merge branch-name                    # Merge branch into current
git merge --abort                        # Abort a merge

# Viewing changes
git diff                                 # See unstaged changes
git log                                  # View commit history
git log --oneline --graph               # Pretty commit history

# Undoing
git checkout .                           # Discard all changes
git reset --hard HEAD                    # Reset to last commit
git revert HEAD                          # Create commit that undoes last commit
```

<br>

## 🎓 Summary

**The golden workflow:**
1. ✅ Always work on a feature branch, never on main
2. ✅ Pull latest changes before starting work
3. ✅ Commit often with clear messages
4. ✅ Merge main into YOUR branch to resolve conflicts
5. ✅ Push your branch and create a Pull Request
6. ✅ After merging, delete the feature branch and start fresh

**Remember:**
- Conflicts are normal and expected
- Communicate with your teammate
- When in doubt, ask before forcing changes
- Your local repository is safe to experiment in
- You can always abort a merge with `git merge --abort`

Good luck with your collaborative project! 🚀