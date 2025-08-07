import multiprocessing
import listener
import event_loop
import server


def run_all_servers():
    processes = []

    p1 = multiprocessing.Process(target=listener.serve_listener)
    p1.start()
    processes.append(p1)

    p2 = multiprocessing.Process(target=event_loop.serve_event_loop)
    p2.start()
    processes.append(p2)

    p3 = multiprocessing.Process(target=server.serve_backend)
    p3.start()
    processes.append(p3)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nShutting down all servers...")
        for p in processes:
            p.terminate()
            p.join()
        print("All servers stopped.")


if __name__ == "__main__":
    run_all_servers()
