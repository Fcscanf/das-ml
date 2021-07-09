import asyncio
import time


async def say_after(delay, what):
    print(what)
    await asyncio.sleep(delay)


async def main():
    print(f"started at {time.strftime('%X')}")
    print("*")
    sa1 = say_after(1, 'hello')
    print("**")
    sa2 = say_after(2, 'world')
    print("***")
    await asyncio.wait([sa1, sa2])
    print(f"finished at {time.strftime('%X')}")


asyncio.run(main())
