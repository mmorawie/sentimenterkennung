

Stat = {
    box = 2,
    maxlen = 80,
    limit = 80/2,
    bucket = nil,
    zeroc = function(self)
        self.cnt = 0
        self.tot = 0
    end,
    zero = function(self)
        self.wordscorrect = 0
        self.phrasescorrect = 0
        self.sentencescorrect = 0
        self.totalcorrect = 0

        self.nwords = 0
        self.nphrases = 0
        self.nsentences = 0
        self.ntotal = 0

        self.lenghts = {}
        self.lenghtscorrect = {}
        for i = 1, 100 do 
            self.lenghts[i] = 0 
            self.lenghtscorrect[i] = 0
        end
    end,
    words = function(self, out, rea)
        if tablebucket(out) == self.bucket(rea) then 
            self.wordscorrect = self.wordscorrect + 1
            self.totalcorrect = self.totalcorrect + 1
        end
        self.nwords = self.nwords + 1
    end,
    sentences = function(self, out, rea)
        if tablebucket(out) == self.bucket(rea) then 
            self.sentencescorrect = self.sentencescorrect + 1
            self.phrasescorrect = self.phrasescorrect + 1
        end
        self.nsentences = self.nsentences + 1
        self.nphrases = self.nphrases + 1
    end,
    phrases = function(self, out, rea, size)
        if tablebucket(out) == self.bucket(rea) then 
            self.phrasescorrect = self.phrasescorrect + 1 
        end
        self.nphrases = self.nphrases + 1
        for i = 1,self.limit do
            if (i-1)*self.box < size and i*self.box >= size then
                self.lenghts[i] = self.lenghts[i] + 1
                if tablebucket(out) == self.bucket(rea) then self.lenghtscorrect[i] = self.lenghtscorrect[i] + 1 end
                break
            end
        end
    end,
    count = function(self, a)
        self.cnt = self.cnt + a;
        self.tot = self.tot + 1;
    end,
    over = function(self)
        result = {}
        result.sentences = self.sentencescorrect/self.nsentences
        result.phrases = self.phrasescorrect/self.nphrases
        result.wordss = self.wordscorrect/self.nwords
        len = {}
        for i = 1, self.limit do
            if self.lenghts[i] == 0 then break end
            len[i] =  self.lenghtscorrect[i]/self.lenghts[i]
        end
        return result, len
    end, 
    printout = function(self)
        print("\n sentences = ", self.sentencescorrect/self.nsentences)
        print("words =     ", self.wordscorrect/self.nwords)
        print("phrases =   ", self.phrasescorrect/self.nphrases)
        io.write("{")
        for i = 1, self.limit do
            if self.lenghts[i] == 0 then break end
            io.write("{" .. i*self.box .. ", " .. self.lenghtscorrect[i]/self.lenghts[i] .. "},\n")
        end
        io.write("};\n")
    end,
    cntstr = function(self)
       return "count = " .. tostring(self.cnt/self.tot)
    end
}





return Stat

