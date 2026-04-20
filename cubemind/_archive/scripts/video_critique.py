"""Video critique demo — CubeMind watches a video and gives its opinion.

Usage:
    python scripts/video_critique.py path/to/video.mp4
    python scripts/video_critique.py path/to/video.mp4 --save-taste cubemind_taste
    python scripts/video_critique.py path/to/video.mp4 --load-taste cubemind_taste

The more videos it watches, the more its taste develops.
"""

import argparse
import os
import sys

from cubemind.perception.scene import SceneAnalyzer


def main():
    parser = argparse.ArgumentParser(description="CubeMind video critique")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--subsample", type=int, default=3, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=600, help="Max frames to process")
    parser.add_argument("--save-taste", type=str, default="cubemind_taste", help="Save taste profile")
    parser.add_argument("--load-taste", type=str, default="cubemind_taste", help="Load taste profile")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}")
        sys.exit(1)

    # Initialize scene analyzer
    analyzer = SceneAnalyzer(
        feature_dim=104,
        snn_neurons=256,
        d_vsa=2048,
    )

    # Load prior taste if available
    if args.load_taste and os.path.exists(f"{args.load_taste}_taste.npz"):
        analyzer.load(args.load_taste)
        print(f"Loaded taste profile: {analyzer.taste.total_scenes} prior scenes")

    # Analyze
    result = analyzer.analyze_video(
        args.video,
        subsample=args.subsample,
        max_frames=args.max_frames,
    )

    # Print critique
    print("\n" + "=" * 60)
    print("  CubeMind Video Critique")
    print("=" * 60)
    print()
    print(result["critique"])
    print()
    print(f"Scenes detected: {result['n_segments']}")
    print(f"Frames processed: {result['frames_processed']}")
    print(f"Neurochemistry: cortisol={result['neurochemistry']['cortisol']:.2f}, "
          f"dopamine={result['neurochemistry']['dopamine']:.2f}, "
          f"serotonin={result['neurochemistry']['serotonin']:.2f}")
    print(f"Total taste experiences: {analyzer.taste.total_scenes}")

    # Save taste for future runs
    if args.save_taste:
        analyzer.save(args.save_taste)
        print(f"\nTaste saved to {args.save_taste}_*")


if __name__ == "__main__":
    main()
