import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
path='6.circle-border'
# Function to interpolate between two dataframes for a smooth transition
def interpolate_dfs(df_start, df_end, alpha):
    df_interp = df_start.add_suffix('_start').merge(df_end.add_suffix('_end'), left_on='ID_start', right_on='ID_end')
    df_interp['X'] = (1 - alpha) * df_interp['X_start'] + alpha * df_interp['X_end']
    df_interp['Y'] = (1 - alpha) * df_interp['Y_start'] + alpha * df_interp['Y_end']
    return df_interp[['ID_start', 'X', 'Y']].rename(columns={'ID_start': 'ID'})

# Load real and fake datasets
real_data_paths = sorted(glob.glob('formation/'+path+'/[0-20].csv'))
fake_data_paths = sorted(glob.glob('formation/'+path+'/fake[0-20].csv'))
real_dfs = [pd.read_csv(path) for path in real_data_paths]
fake_dfs = [pd.read_csv(path) for path in fake_data_paths]
dot_size=10
# Load condition data
cond_df = pd.read_csv('formation/'+path+'/cond.csv')

# Set up the figure for animation
fig, (ax1, ax_cond, ax2) = plt.subplots(1, 3, figsize=(15, 5))
plt.close()  # Prevents the initial static plot from displaying

# Plot condition data statically in the center subplot
ax_cond.scatter(cond_df['X'], -cond_df['Y'], s=dot_size, color='green')  # Negate Y for correct orientation
ax_cond.set_axis_off()
ax_cond.text(0.5, 1.05, f'Condition (n={len(cond_df)})', ha='center', transform=ax_cond.transAxes, fontsize=12)


# Define the animation update function
def animate_with_counts(i):
    alpha = (i % num_frames) / num_frames  # Current interpolation state (0 to 1)
    step = min(i // num_frames, len(real_dfs) - 2)  # Current step in the dataset transitions

    # Interpolate the real and fake dataframes
    real_interp = interpolate_dfs(real_dfs[step], real_dfs[step + 1], alpha)
    fake_interp = interpolate_dfs(fake_dfs[step], fake_dfs[step + 1], alpha)

    # Update real plot
    ax1.clear()
    ax1.scatter(real_interp['X'], -real_interp['Y'], s=dot_size, color='blue')  # Negate Y for correct orientation
    ax1.text(0.5, 1.05, f'Real (n={len(real_interp)})', ha='center', transform=ax1.transAxes, fontsize=12)
    ax1.set_axis_off()

    # Update fake plot
    ax2.clear()
    ax2.scatter(fake_interp['X'], -fake_interp['Y'], s=dot_size, color='red')  # Negate Y for correct orientation
    ax2.text(0.5, 1.05, f'Generated (n={len(fake_interp)})', ha='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_axis_off()

    return ax1, ax_cond, ax2,

# Number of frames per transition
num_frames = 20

# Create the animation
ani_with_counts = animation.FuncAnimation(fig, animate_with_counts, frames=num_frames * (len(real_dfs) - 1), interval=100, blit=False)

# Save the animation as a GIF
counts_gif_path = 'formation/'+path+'/'+path+'.gif'
ani_with_counts.save(counts_gif_path, writer='pillow', fps=10)

# The path to the saved GIF
print(counts_gif_path)
