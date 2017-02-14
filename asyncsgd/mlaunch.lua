-- mpi launch
-- Author: Sixin Zhang (zsx@cims.nyu.edu)
-- Author: Umhau (umhau@alum.gcc.edu)
-- mpirun -n 12 luajit mlaunch.lua

--[[ NOTES --------------------------------------------------------------------

This script is used to launch mpi.  The user's script goes at the bottom, where
goot.lua has been referenced.   Edit this file only to change the variables 
noted.

MPI is configured so this script will be running separately on each available 
core on each available machine in the cluster - so the 'ranks' below will range
from 0-7 if there are two machines with 4 CPU cores each.  I don't have any 
GPUs, so I can't speak to how those are presented.

It is assumed the EAMSGD optimizer is desired.

--]]

-- VARIABLES ------------------------------------------------------------------

local oncuda = false -- Set for working with CPUs. Change this if using GPUs.
local torchfile = 'goot.lua' -- name of torch file to run with MPI
local iterations = 10 -- i.e., epochs.  don't need that many for testing.

-- there's other EAMSGD variables that can be tuned below. I'll do that later.

-- GPU SETTINGS ---------------------------------------------------------------

local AGPU = nil
if oncuda then
   require 'cutorch'
   AGPU = {1,2,3,4,5,6} -- use the first 6 gpus on each machine
end

local gpuid = -1

-- MPI CONFIGURATION ----------------------------------------------------------

dofile('init.lua')
mpiT.Init()

local world = mpiT.COMM_WORLD
local rank = mpiT.get_rank(world)
local size = mpiT.get_size(world)

local conf = {}
conf.rank = rank
conf.world = world
conf.sranks = {}
conf.cranks = {}
for i = 0,size-1 do
   if math.fmod(i,2)==0 then
      table.insert(conf.sranks,i)
   else
      table.insert(conf.cranks,i)
   end
end

opt = {}
opt.name = 'eamsgd' -- using most efficient optimizer
opt.su = 100
opt.mva = 0.9/6 -- this is \beta/p when p=6
opt.lr = 1e-2 -- (or 1e-1) order of magnitude from the other - what's the difference?
opt.mom = 0.99

opt.maxepoch = iterations

-- determine if the current node should be server or client. Seems like there
-- should be more clients than servers...investigate later.  (change the '2'?)
if math.fmod(rank,2)==0 then
   -- if the rank # is even, it's a server
   print('[server] rank',rank,'use cpu')
   torch.setdefaulttensortype('torch.FloatTensor')  
   local ps = pServer(conf)
   ps:start()

else
   -- if node rank # is odd, it's a client.  This means we have to choose how 
   -- to process the metric $#!7-ton of data that's going to be directed this 
   -- way.  So, this is where we configure our GPUs or CPUs.  

   if AGPU then
      -- if not nil, GPUs are enabled
      require 'cunn'
      -- use CUDA
      local gpus = cutorch.getDeviceCount()
      -- how many GPUs available on this machine?
      gpuid = AGPU[(rank%(size/2)) % gpus + 1]
      -- use the node's rank to set the ID of each(?) GPU
      cutorch.setDevice(gpuid)
      print('[client] rank ' .. rank .. ' use gpu ' .. gpuid)
      torch.setdefaulttensortype('torch.CudaTensor')

   else
      -- if the GPU flag is set FALSE, we're using CPUs
      print('[client] rank ' .. rank .. ' use cpu')
      torch.setdefaulttensortype('torch.FloatTensor')
   end

   -- done with configuring the processors.  These are settings specific to the
   -- node at hand, now that we know exactly what it's going to be doing.
   opt.gpuid = gpuid       -- Tell the optimizer if GPUs are available.
   opt.pc = pClient(conf)  -- MPI settings for communicating with the other nodes.
   opt.rank = rank         -- Simple access to the node number.  

   -- Time to run the training algorithm.  This is not an arbitrary script,
   -- and must contain some cruicial settings.
   dofile(torchfile)

end

-- clean up the MPI communication channels.
mpiT.Finalize()
