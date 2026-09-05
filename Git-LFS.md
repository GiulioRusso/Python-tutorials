<center><h1>📦 Git LFS Guide</h1></center>

A beginner-friendly guide to **Git Large File Storage (Git LFS)**: what it is, why plain Git struggles with big files, and how to version datasets, model weights, and other large binaries without bloating your repository. **No prior knowledge is assumed** beyond basic Git (`clone`, `add`, `commit`, `push`).

<br>

## 1️⃣ The Problem Git LFS Solves

Git was built for **text**. It stores the full history of every file forever, and it is very good at compressing and diffing source code.

Binary files break this model:

- A 200 MB model checkpoint **cannot be diffed**. Change one weight, and Git stores another full 200 MB copy.
- Every clone downloads **the entire history**. Ten versions of a 200 MB file means every collaborator downloads 2 GB, even if they only need the latest.
- Hosting platforms reject or warn on large files. GitHub **blocks any file over 100 MB** and warns above 50 MB.
- The `.git` folder grows without bound and never shrinks.

```
Plain Git with a 200 MB dataset, edited 5 times
────────────────────────────────────────────────
working copy      200 MB
.git history      ~1 GB   ← every version, forever, in every clone
```

**Git LFS** replaces the large file in your repository with a tiny text **pointer**, and stores the actual file content on a separate LFS server. Git history stays small; big files are downloaded only when needed.

```
Git + LFS with the same dataset
────────────────────────────────────────────────
working copy      200 MB
.git history      ~5 KB   ← just pointers
LFS store         downloaded on demand, only versions you check out
```

<br>

## 2️⃣ How Git LFS Works

Git LFS installs **filters** that run automatically on `git add` and `git checkout`.

| Step | What happens |
|------|--------------|
| `git add data.zip` | LFS **clean** filter intercepts the file, uploads-nothing-yet but computes its SHA-256, writes a **pointer file** into the Git index. |
| `git commit` | The commit records the pointer, not the binary. |
| `git push` | The pointer goes to the normal Git remote; the **binary is uploaded to the LFS store**. |
| `git checkout` / `git pull` | LFS **smudge** filter sees the pointer and downloads the real file from the LFS store, replacing the pointer in your working directory. |

A pointer file looks like this — this is what is actually committed to Git:

```
version https://git-lfs.github.com/spec/v1
oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393
size 209715200
```

The pointer names the object by hash and size. The `.gitattributes` file tells Git **which paths** to run the filter on.

<br>

## 3️⃣ Installation

### macOS

```bash
brew install git-lfs
```

### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install git-lfs
```

### Windows

Git LFS ships with **Git for Windows**. Otherwise download from <https://git-lfs.com>.

### One-time per-machine setup

After installing, run once per user account:

```bash
git lfs install
```

This adds the LFS filter config to your global `~/.gitconfig`. Verify:

```bash
git lfs version
# git-lfs/3.5.1 (GitHub; darwin arm64; go 1.21.5)
```

<br>

## 4️⃣ Setting Up a Repository

Inside an existing Git repository:

```bash
# Enable LFS hooks for this repo (safe to run even if already global)
git lfs install
```

That is all. The next step — telling LFS which files to manage — is done with `git lfs track`.

<br>

## 5️⃣ Tracking Files

You track files **by pattern**, not one by one. Patterns are stored in `.gitattributes`.

```bash
# By extension (most common)
git lfs track "*.pth"
git lfs track "*.h5"
git lfs track "*.zip"
git lfs track "*.parquet"

# A whole directory
git lfs track "data/**"

# A specific large file
git lfs track "models/resnet50_final.pt"
```

Each command appends a line to `.gitattributes`:

```
*.pth filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
data/** filter=lfs diff=lfs merge=lfs -text
```

**Commit `.gitattributes` immediately** and before adding the large files:

```bash
git add .gitattributes
git commit -m "Track model and data files with Git LFS"
```

> ⚠️ **Order matters.** `git lfs track` only affects files added **after** the pattern is in `.gitattributes`. Files already committed to Git normally stay in normal Git until you [migrate](#8️⃣-migrating-existing-large-files) them.

### Checking what is tracked

```bash
git lfs track          # list patterns
git lfs ls-files       # list files currently managed by LFS in HEAD
```

<br>

## 6️⃣ Daily Workflow

Once `.gitattributes` is set, **your workflow does not change**:

```bash
cp ~/Downloads/dataset.zip data/
git add data/dataset.zip
git commit -m "Add training dataset v1"
git push
```

Behind the scenes, `push` uploads `dataset.zip` to the LFS store and only the pointer to the Git remote.

Confirm a file is a pointer, not a real blob, in the committed tree:

```bash
git show HEAD:data/dataset.zip
# version https://git-lfs.github.com/spec/v1
# oid sha256:...
# size ...
```

Check push status and what would be uploaded:

```bash
git lfs status
git lfs push origin main --dry-run
```

<br>

## 7️⃣ Cloning and Pulling LFS Repositories

A normal `git clone` **also fetches LFS files** for the checked-out commit, provided Git LFS is installed on that machine.

```bash
git lfs install          # once per machine — do not forget this
git clone https://github.com/user/project.git
```

If a collaborator forgot `git lfs install`, they get **pointer files instead of real data**. Fix:

```bash
git lfs install
git lfs pull
```

### Useful control over what gets downloaded

```bash
# Clone without downloading any LFS blobs (fast, pointers only)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/user/project.git

# Then fetch only what you need
git lfs pull --include="data/train/**" --exclude="data/raw/**"
```

`git lfs fetch` downloads to the local LFS cache without touching the working tree; `git lfs checkout` then materializes the files. `git lfs pull` = `fetch` + `checkout`.

<br>

## 8️⃣ Migrating Existing Large Files

If large files are **already in your Git history**, adding an LFS pattern does not remove them from past commits — the repo stays bloated. Use `git lfs migrate`.

```bash
# See which file types dominate history
git lfs migrate info --everything

# Rewrite ALL history, moving *.pth and *.zip into LFS
git lfs migrate import --everything --include="*.pth,*.zip"
```

> ⚠️ **This rewrites history.** Every commit hash changes. Coordinate with collaborators, then force-push:
>
> ```bash
> git push --force --all
> git push --force --tags
> ```
>
> Everyone else must re-clone or hard-reset. Do this on a quiet branch/day.

To go the other way (pull files back out of LFS):

```bash
git lfs migrate export --everything --include="*.zip"
```

<br>

## 9️⃣ Storage, Quotas, and Hosting

The LFS store is provided by your Git host, and it is usually **metered separately** from normal repo storage.

| Host | Free LFS storage | Free bandwidth / month | Notes |
|------|------------------|------------------------|-------|
| GitHub | 1 GB | 1 GB | Extra sold in 50 GB "data packs". Bandwidth counts every clone/pull. |
| GitLab | 10 GB (shared project storage) | included in transfer limits | Self-hosted GitLab: unlimited, your disk. |
| Bitbucket | 1 GB | unlimited | Raises to 5–10 GB on paid plans. |
| Hugging Face Hub | large, dataset-oriented | generous | Purpose-built for models/datasets; uses Git LFS natively. |

**Bandwidth is the trap.** A 5 GB dataset pulled by 10 people twice a month = 100 GB of egress. For heavy dataset sharing, consider [alternatives](#-alternatives-to-git-lfs) or a self-hosted LFS server.

Self-hosted LFS options: GitLab, Gitea, or a standalone server like [`giftless`](https://github.com/datopian/giftless) backed by S3.

<br>

## 🔟 Common Errors and How to Fix Them

### "This repository is over its data quota"

You hit the LFS storage or bandwidth limit. Buy a data pack, delete old LFS objects, or move the data off LFS.

### Files show up as pointer text after clone

Git LFS was not installed when cloning.

```bash
git lfs install
git lfs pull
```

### `git lfs track` seems to do nothing

- Did you `git add .gitattributes` and commit it?
- The file was **already committed** before tracking — it needs `git lfs migrate`.
- Pattern quoting: always quote the pattern so the **shell does not expand it** — `git lfs track "*.zip"`, not `git lfs track *.zip`.

### Push rejected: "File is 143 MB; this exceeds GitHub's file size limit of 100 MB"

The file went into **normal Git**, not LFS, because the pattern was added too late.

```bash
git lfs migrate import --include="*.bin" --include-ref=refs/heads/my-branch
git push --force
```

### `smudge filter lfs failed`

Network issue, dead LFS server, or auth failure during checkout. Retry:

```bash
git lfs fetch --all
git checkout .
```

### Want to see how much space LFS is using locally

```bash
git lfs env          # shows local and remote LFS paths
du -sh .git/lfs      # size of the local LFS cache
git lfs prune        # delete local LFS files no longer referenced by recent commits
```

<br>

## 🧭 Best Practices

1. ✅ **Install `git lfs install` on every machine** — CI runners included.
2. ✅ **Commit `.gitattributes` first**, before the large files it covers.
3. ✅ Track **by extension**, not individual files, so new data is covered automatically.
4. ✅ Keep raw datasets that **never change** out of Git entirely (see alternatives) — LFS shines for files that evolve and need versioning.
5. ✅ In CI, use `actions/checkout` with `lfs: true`, or `GIT_LFS_SKIP_SMUDGE=1` plus a selective `git lfs pull`.
6. ✅ Run `git lfs prune` periodically to reclaim local disk.
7. ✅ Document in your README that the repo uses LFS and that collaborators must install it.
8. ✅ Watch **bandwidth**, not just storage — every clone of a large history costs egress.
9. ❌ Do not `git lfs migrate` a shared branch without warning everyone — it rewrites history.

<br>

## 🔀 Alternatives to Git LFS

Git LFS is the right tool when large files **change over time and must be versioned alongside code**. When that is not the case:

| Tool | Best for |
|------|----------|
| **DVC** (`dvc`) | ML datasets and pipelines; stores data in S3/GCS/SSH, keeps lightweight `.dvc` pointers in Git. Better than LFS for large, rarely-edited datasets and reproducible pipelines. |
| **Hugging Face Hub** | Publishing models and datasets; free generous storage, built on Git LFS with a nicer UI and `huggingface_hub` library. |
| **`git-annex`** | Fully decentralized large-file management; steeper learning curve. |
| **Plain object storage** (S3, MinIO, rsync) | Static datasets referenced by a download script or checksum file. Simplest and cheapest. |
| **`rclone` + a manifest** | Syncing large data directories to cloud storage with a checksum manifest committed to Git. |

Rule of thumb: **code and small evolving binaries → Git LFS. Big static datasets → DVC or object storage.**

<br>

## 📋 Cheat Sheet

```bash
# --- Setup (once per machine) ---
git lfs install

# --- Track files (per repo) ---
git lfs track "*.pth"              # add pattern to .gitattributes
git lfs track "data/**"
git add .gitattributes
git commit -m "Configure Git LFS"

# --- Normal workflow (unchanged) ---
git add big_file.zip
git commit -m "Add data"
git push

# --- Inspect ---
git lfs track                      # list patterns
git lfs ls-files                   # files managed by LFS
git lfs status                     # staged LFS changes
git lfs env                        # config, paths, endpoints
du -sh .git/lfs                    # local cache size

# --- Clone / pull ---
git lfs install && git clone <url>
git lfs pull                       # fetch + checkout LFS files
GIT_LFS_SKIP_SMUDGE=1 git clone <url>   # pointers only
git lfs pull --include="data/train/**"  # selective

# --- Fetch control ---
git lfs fetch --all               # download all LFS history to cache
git lfs checkout                  # materialize pointers in working tree
git lfs prune                     # drop unreferenced local LFS files

# --- Migrate existing history ---
git lfs migrate info --everything
git lfs migrate import --everything --include="*.pth,*.zip"
git push --force --all            # ⚠️ rewrites history

# --- Undo: move files out of LFS ---
git lfs migrate export --everything --include="*.zip"
```

<br>

## 🎓 Summary

1. ✅ Git LFS swaps large files for tiny **pointer files**; real content lives on an **LFS store**.
2. ✅ `.gitattributes` decides what LFS manages — **commit it first**.
3. ✅ Everyday `add` / `commit` / `push` / `clone` commands stay the same.
4. ✅ Files already in history need `git lfs migrate` (rewrites history — coordinate).
5. ✅ Mind the **bandwidth quota**, not just storage.
6. ✅ Big **static** datasets often belong in **DVC or object storage**, not LFS.

If your repo clones cleanly and the model weights are just there, LFS is working. 🚀

<br>
<br>
<br>

[Back to Index 🗂️](./README.md)
