import argparse
import zadanie_1, zadanie_2, zadanie_3, zadanie_4, zadanie_5, zadanie_6, zadanie_7


def main():
    parser = argparse.ArgumentParser(description="Image Processing Homework Runner")
    parser.add_argument(
        "--image", type=str, default="example.jpg", help="Path to image"
    )
    parser.add_argument(
        "--task", type=int, default=0, help="Task number (1-7) or 0 for all"
    )
    args = parser.parse_args()

    modules = {
        1: zadanie_1,
        2: zadanie_2,
        3: zadanie_3,
        4: zadanie_4,
        5: zadanie_5,
        6: zadanie_6,
        7: zadanie_7,
    }

    if args.task in modules:
        modules[args.task].run(args.image)
    else:
        for idx, mod in modules.items():
            print(f"\n--- Running Task {idx} ---")
            try:
                mod.run(args.image)
            except Exception as e:
                print(f"Error on Task {idx}: {e}")


if __name__ == "__main__":
    main()
