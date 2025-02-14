# Rockfish readme

1. Login to rockfish

```bash
ssh {YOUR ROCKFISH USERNAME}@login.rockfish.jhu.edu
```

2. Go to the working directory

```bash
cd scr4_struelo1/flepimop-code/twallema
```

3. Installation (do once)

3a. Clone the Github repository (do once)

```bash
git clone git@github.com:twallema/influenza-USA.git
```

3b. Install the conda environment (do once)

```bash
module load anaconda3
conda env create -f influenza_USA_env.yml
```

3c. Install model package inside conda environment

```bash
conda activate INFLUENZA-USA
pip install -e .
```

4. Setup 

4a. Checkout to the right branch

```bash
cd influenza-USA
git branch 
git checkout <my_branch>
```

4b. Update branch

```bash
git checkout origin
git pull
```

5. Submit the job to the cluster

5a. Write a job submission script

See examples in this Github repo.

5b. Make sure it's executable

```bash
chmod +x my_submision_script.sh
```

5c. Submit the script to `slurm`

```bash
sbatch my_script.sh
```

5d. Monitor your job

```bash
squeue -u your_username
scancel --name=your_job_name
```

# Tips and Tricks

- Reset all changes made on cluster in git:

```bash
git reset --hard && git clean -f -d
```

- Copy from the HPC to local computer.

    - Open a terminal where you want to place the files on your computer.
    - Run

    ```bash
    scp -r <username>@rfdtn1.rockfish.jhu.edu:/home/<username>/.ssh/<key_name.pub> .
    ```