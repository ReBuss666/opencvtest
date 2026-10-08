import os
import argparse
import matplotlib.pyplot as plt

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

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    modules = {
        1: zadanie_1,
        2: zadanie_2,
        3: zadanie_3,
        4: zadanie_4,
        5: zadanie_5,
        6: zadanie_6,
        7: zadanie_7,
    }

    def execute_task(task_num, module):
        print(f"\n--- Running Task {task_num} ---")
        save_path = os.path.join(output_dir, f"result_task_{task_num}.png")

        def custom_show():
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f" [✓] Сохранено: {save_path}")
            original_show()

        original_show = plt.show
        plt.show = custom_show

        try:
            module.run(args.image)
        finally:
            plt.show = original_show

    if args.task in modules:
        execute_task(args.task, modules[args.task])
    else:
        for idx, mod in modules.items():
            try:
                execute_task(idx, mod)
            except Exception as e:
                print(f"Error on Task {idx}: {e}")


if __name__ == "__main__":
    main()
