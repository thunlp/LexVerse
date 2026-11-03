from lexverse.agents.native import NativeRunner
from lexverse.agents.static import run


class SimuCourtRunner(NativeRunner):
    async def run(self, task, work_dir):
        return await run(task, self.connections["default"])
