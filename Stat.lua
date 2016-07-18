

Stat = {
    box = 2,
    maxlen = 80,
    limit = 80/2,
    zero = function(self)
        self.wordscorrect = 0
        self.phrasescorrect = 0
        self.sentencescorrect = 0
        self.totalcorrect = 0

        self.nwords = 0
        self.nphrases = 0
        self.nsentences = 0
        self.ntotal = 0

        self.wordserror = 0
        self.phraseserror = 0
        self.sentenceserror = 0
        self.totalerror = 0

        self.lenghts = {}
        self.lenghtscorrect = {}
        for i = 1, 100 do 
            self.lenghts[i] = 0 
            self.lenghtscorrect[i] = 0
        end
    end,
    words = function(self, out, rea)
        if bucket(out) == bucket(rea) then 
            self.wordscorrect = self.wordscorrect + 1
            self.totalcorrect = self.totalcorrect + 1
        end
        self.wordserror = self.wordserror + math.abs(out - rea)
        self.nwords = self.nwords + 1
    end,
    sentences = function(self, out, rea)
        if bucket(out) == bucket(rea) then 
            self.sentencescorrect = self.sentencescorrect + 1
            self.phrasescorrect = self.phrasescorrect + 1
        end
        self.sentenceserror = self.sentenceserror + math.abs(out - rea)
        self.phraseserror = self.phraseserror + math.abs(out - rea)
        self.nsentences = self.nsentences + 1
        self.nphrases = self.nphrases + 1
    end,
    phrases = function(self, out, rea, size)
        if bucket(out) == bucket(rea) then self.phrasescorrect = self.phrasescorrect + 1 end
        self.phraseserror = self.phraseserror + math.abs(out - rea)
        self.nphrases = self.nphrases + 1
        for i = 1,self.limit do
            if (i-1)*self.box < size and i*self.box >= size then
                self.lenghts[i] = self.lenghts[i] + 1
                if bucket(out) == bucket(rea) then self.lenghtscorrect[i] = self.lenghtscorrect[i] + 1 end
                break
            end
        end
    end,
    printout = function(self)
        print("sentences = ", self.sentencescorrect/self.nsentences)
        --print("words =     ", self.wordscorrect/self.nwords)
        print("phrases =   ", self.phrasescorrect/self.nphrases)
        io.write("{")
        for i = 1, self.limit do
            if self.lenghts[i] == 0 then break end
            io.write("{" .. i*self.box .. ", " .. self.lenghtscorrect[i]/self.lenghts[i] .. "},\n")
        end
        io.write("};\n")
    end
}





return Stat

