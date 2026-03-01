#!/usr/bin/env python3
"""Generate dynamic folding animation showing protein folding in real-time."""
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Tuple
import tempfile

sys.path.append(str(Path(__file__).parent.parent))

from model.architecture import EvolvableProteinFoldingModel
from model.data_generator import ProteinDataGenerator

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio not available for animation. Install with: pip install imageio")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation, PillowWriter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available for animation")

class FoldingAnimator:
    """Create dynamic animations of protein folding process."""
    
    def __init__(self, model_path: str):
        self.model = EvolvableProteinFoldingModel.load_checkpoint(model_path)
        self.model.eval()
        self.data_generator = ProteinDataGenerator()
    
    def generate_folding_trajectory(self, sequence: str, n_steps: int = 50) -> List[Dict]:
        """Generate intermediate folding states from unfolded to folded.
        
        This simulates the folding process by:
        1. Starting with extended chain
        2. Gradually morphing towards predicted structure
        3. Using confidence to control folding speed per residue
        """
        # Get final predicted structure
        seq_tensor = torch.zeros(1, len(sequence), dtype=torch.long)
        for i, aa in enumerate(sequence):
            if aa in self.data_generator.AMINO_ACID_TO_IDX:
                seq_tensor[0, i] = self.data_generator.AMINO_ACID_TO_IDX[aa]
        
        with torch.no_grad():
            predictions = self.model(seq_tensor)
        
        final_coords = predictions['coordinates'][0].numpy()
        confidence = predictions['confidence'][0].numpy()
        
        # Generate initial extended structure
        initial_coords = np.zeros_like(final_coords)
        for i in range(len(sequence)):
            initial_coords[i] = np.array([i * 3.8, 0, 0])  # Extended chain
        
        # Generate intermediate frames
        trajectory = []
        
        for step in range(n_steps + 1):
            progress = step / n_steps
            
            # Residue-specific folding progress based on confidence
            # High confidence residues fold faster
            residue_progress = np.minimum(
                progress * (1 + confidence),  # Confident regions fold faster
                1.0
            )
            
            # Interpolate coordinates
            frame_coords = (
                initial_coords * (1 - residue_progress[:, np.newaxis]) +
                final_coords * residue_progress[:, np.newaxis]
            )
            
            # Add some random fluctuation for realism (decreases as folding progresses)
            noise_scale = 0.5 * (1 - progress)
            frame_coords += np.random.randn(*frame_coords.shape) * noise_scale
            
            trajectory.append({
                'coordinates': frame_coords,
                'confidence': confidence,
                'progress': progress,
                'sequence': sequence
            })
        
        return trajectory
    
    def create_matplotlib_animation(self, trajectory: List[Dict], output_path: str,
                                   fps: int = 10, show_confidence: bool = True):
        """Create animation using matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for animation")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up plot limits
        all_coords = np.concatenate([frame['coordinates'] for frame in trajectory])
        margin = 5
        ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
        ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
        ax.set_zlim(all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin)
        
        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        ax.set_zlabel('Z (Å)', fontsize=12)
        
        # Initialize plot elements
        scatter = ax.scatter([], [], [], c=[], cmap='RdYlGn', s=50, 
                           vmin=0, vmax=1, alpha=0.8)
        line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.5)
        title = ax.set_title('', fontsize=14, fontweight='bold')
        
        # Colorbar for confidence
        if show_confidence:
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.1)
            cbar.set_label('Prediction Confidence', fontsize=12)
        
        def init():
            scatter._offsets3d = ([], [], [])
            line.set_data([], [])
            line.set_3d_properties([])
            return scatter, line, title
        
        def update(frame_idx):
            frame = trajectory[frame_idx]
            coords = frame['coordinates']
            confidence = frame['confidence']
            progress = frame['progress']
            
            # Update scatter plot
            scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
            
            if show_confidence:
                scatter.set_array(confidence)
            
            # Update backbone line
            line.set_data(coords[:, 0], coords[:, 1])
            line.set_3d_properties(coords[:, 2])
            
            # Update title
            title.set_text(
                f'Protein Folding Animation\n'
                f'Progress: {progress*100:.1f}% | '
                f'Frame {frame_idx+1}/{len(trajectory)}'
            )
            
            # Rotate view slightly for dynamic effect
            ax.view_init(elev=20, azim=frame_idx * 2)
            
            return scatter, line, title
        
        print(f"Generating animation with {len(trajectory)} frames...")
        anim = FuncAnimation(fig, update, init_func=init, 
                           frames=len(trajectory), interval=1000/fps,
                           blit=False, repeat=True)
        
        # Save animation
        print(f"Saving animation to {output_path}...")
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=100)
        
        plt.close()
        print(f"✓ Animation saved: {output_path}")
    
    def create_frame_sequence(self, trajectory: List[Dict], output_dir: str,
                             show_confidence: bool = True):
        """Create sequence of frames as individual images."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {len(trajectory)} frames...")
        
        for frame_idx, frame in enumerate(trajectory):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            coords = frame['coordinates']
            confidence = frame['confidence']
            progress = frame['progress']
            
            # Plot structure
            if show_confidence:
                scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                                   c=confidence, cmap='RdYlGn', s=50, 
                                   vmin=0, vmax=1, alpha=0.8)
                plt.colorbar(scatter, ax=ax, label='Confidence')
            else:
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                         c='blue', s=50, alpha=0.8)
            
            # Plot backbone
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                   'b-', linewidth=2, alpha=0.5)
            
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_title(f'Folding Progress: {progress*100:.1f}%', fontweight='bold')
            
            # Set consistent view angle
            ax.view_init(elev=20, azim=frame_idx * 2)
            
            # Save frame
            output_file = output_dir / f'frame_{frame_idx:04d}.png'
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            if (frame_idx + 1) % 10 == 0:
                print(f"  Generated {frame_idx + 1}/{len(trajectory)} frames")
        
        print(f"\n✓ All frames saved to {output_dir}")
        
        # Create GIF if imageio available
        if IMAGEIO_AVAILABLE:
            gif_path = output_dir.parent / 'folding_animation.gif'
            self.create_gif_from_frames(output_dir, gif_path)
    
    def create_gif_from_frames(self, frames_dir: Path, output_path: str, fps: int = 10):
        """Create GIF from frame sequence."""
        if not IMAGEIO_AVAILABLE:
            return
        
        frames = sorted(frames_dir.glob('frame_*.png'))
        
        if not frames:
            print("No frames found for GIF creation")
            return
        
        print(f"Creating GIF from {len(frames)} frames...")
        
        images = [imageio.imread(str(frame)) for frame in frames]
        imageio.mimsave(output_path, images, fps=fps, loop=0)
        
        print(f"✓ GIF saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Animate protein folding process')
    parser.add_argument('--checkpoint', type=str, default='weights/latest.pt',
                       help='Model checkpoint path')
    parser.add_argument('--sequence', type=str,
                       help='Protein sequence to animate')
    parser.add_argument('--output-dir', type=str, default='animations',
                       help='Output directory for animations')
    parser.add_argument('--n-steps', type=int, default=50,
                       help='Number of folding steps to generate')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for animation')
    parser.add_argument('--format', type=str, default='gif',
                       choices=['gif', 'mp4', 'frames'],
                       help='Output format')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Use example sequence if none provided
    if not args.sequence:
        print("No sequence provided, using example sequence...")
        args.sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLS"
    
    print(f"\nInitializing folding animator...")
    animator = FoldingAnimator(args.checkpoint)
    
    print(f"Generating folding trajectory with {args.n_steps} steps...")
    trajectory = animator.generate_folding_trajectory(args.sequence, args.n_steps)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'gif':
        output_path = output_dir / 'folding_animation.gif'
        animator.create_matplotlib_animation(trajectory, str(output_path), fps=args.fps)
    elif args.format == 'mp4':
        output_path = output_dir / 'folding_animation.mp4'
        print("MP4 format requires ffmpeg. Falling back to GIF...")
        animator.create_matplotlib_animation(trajectory, 
                                           str(output_dir / 'folding_animation.gif'), 
                                           fps=args.fps)
    elif args.format == 'frames':
        animator.create_frame_sequence(trajectory, str(output_dir / 'frames'))
    
    print(f"\n{'='*60}")
    print("Animation Summary:")
    print(f"{'='*60}")
    print(f"  Sequence length: {len(args.sequence)} residues")
    print(f"  Number of frames: {len(trajectory)}")
    print(f"  Animation duration: {len(trajectory)/args.fps:.1f} seconds")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
