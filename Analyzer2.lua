require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'xlua' 
require 'sys' 
require 'lfs' 
require 'penlight'
Double = require 'Double'
Node = require 'Node'


local Analyzer = {}

function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

function loadsets()
	set 		= SST.loadSplit()
	results 	= SST.loadResults()
	sentences 	= SST.loadSentences()
	phrases 	= SST.loadPhrases()
	trees 		= SST.loadTrees()
	testset = {}; a = 1;
	trainset = {}; b = 1;
	devset = {}; c = 1;
	for i = 1, #sentences do
		if set[i] == "1" then trainset[a] = i; a = a + 1 end
		if set[i] == "2" then testset[b] = i; b = b + 1 end
		if set[i] == "3" then devset[c] = i; c = c + 1 end
	end
end

function load(number)
	SST 		= require 'SST'
	Glove		= require 'Glove'
	if number == 0 then
		dictionary = Glove.loadDictionary('./glove.6B/glove.6B.50d.txt',true)
	else
		dictionary = Glove.loadDictionary('./glove.6B/glove.6B.50d.txt',true, number)
	end
	loadsets()
	vectorSize = #(dictionary["do"]) -- vector size
end

function zload()
	SST 		= require 'SST'
	Glove		= require 'Glove'
	loadsets()
	dictionary = {}
	for i = 1, #sentences do
		local wrd = sentences[i]:split('|')
		for j = 1, #wrd do dictionary[ wrd[j] ] = -1 end
	end
	dictionary = Glove.lookup('./glove.6B/glove.6B.50d.txt')
	vectorSize = #(dictionary["do"]) -- vector size
end

function dload()
	SST 		= require 'SST'
	Glove		= require 'Glove'
	--Glove.dset(300)
	vectorSize = 300
	dictionary = {}
	loadsets()
	vectorSize = 300
end

function sent(str, name)
	local par = require 'Parser'
	local res = par.parse(str)
	local wor = res:split("@@")[2]:split("|")
	local tre = map(tonumber, res:split("@@")[1]:split("|") );
	local no = Node.setup(wor, tre)
	local network = torch.load("./networks/" .. name)
	network:forward(no[#no])
	no[#no]:printout()
end

function score(set, mod)
	local no
	local sc = 0
	local loss = 0
	for i = 1, #set do
		mod.bTree:evaluate()
		no = Node.setup(sentences[set[i]]:split('|'), trees[set[i]])
		_, loss = mod.bTree:forward(no[#no])
		--print(loss)
		--if ( tablebucket( no[#no].output ) -  bucket( no[#no].correct ) ) then sc = sc + 1 end
		sc = sc + loss
		mod.bTree:clean(no[#no])
	end
	return sc
end

include('Trainer.lua')
include('bTree.lua')
include('BinTree.lua')
GRU4 = require 'GRU4'
statistics = require 'Stat'

classnumber = 5
require 'Serv'

dload()
load(1000000)


sent('Ala has a stupid cat.', 'module4')
--sent('Ala has a completely stupid cat.', 'module2')
--sent('It is difficult for the isolated individual to work himself out of the immaturity which has become almost natural for him.', 'module2')



return Analyzer
