from Fireworks import Pipe

class VariableBatchGenerator(Pipe):
    """ Generator for variable size batches. """

    def precompute(self):
        """ Precomputes batch indices. """
        start = 0
        end = 1
        n = len(self.dataset)
        indices = [start]
        while end < n:
            if too_big(self.dataset, start, end):
                # Cut off the batch and start a new batch
                indices.append(end)
                start = end
                end += 1
            else:
                # More can fit in memory, so keep going
                end += 1
        indices.append(n)
        self.indices = indices

    def create_generator(self):
        """ Creates a generator that yields variable length batches based on precomputed criteria. """

        def generator():
            start_index = self.indices[0]
            for end_index in self.indices[1:]:
                batch = self.dataset[start_index:end_index]
                batch = pad(batch, self.dataset.embeddings_dict)
                batch = self.to_variable(batch)
                yield batch
                start_index = end_index

        return generator