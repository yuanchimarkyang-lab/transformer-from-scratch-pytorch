from torch.nn.utils.rnn import pad_sequence





def collate_fn(dataBatch, en_sp, de_sp):

    tgt = en_sp.encode([data["en"] for data in dataBatch], out_type=int)
    src = de_sp.encode([data["de"] for data in dataBatch], out_type=int)
    

    encoder_input = []
    decoder_input = []
    decoder_output = []
    
    for data in dataBatch:
        tgt = en_sp.encode(data["en"], out_type=int)
        src = de_sp.encode(data["de"], out_type=int)
        

        encoder_input.append(torch.tensor(src + [EOS], dtype=torch.long))
        decoder_input.append(torch.tensor([BOS] + tgt, dtype=torch.long))
        decoder_output.append(torch.tensor(tgt + [EOS], dtype=torch.long))

    encoder_input = pad_sequence(encoder_input,batch_first=True,padding_value=PAD)
    decoder_input = pad_sequence(decoder_input,batch_first=True,padding_value=PAD)
    decoder_output = pad_sequence(decoder_output,batch_first=True,padding_value=PAD)
    

    return encoder_input, decoder_input, decoder_output