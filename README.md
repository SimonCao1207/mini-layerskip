# mini-layerskip

            [Input Tokens]
                  │
                  ▼
         [Transformer Layers 1 ... k]  <-- Draft Stage (exit_layer = k)
                  │
          ┌────[Draft Tokens]────┐
          │                     │
          │   (Draft Decoding)  │
          ▼                     │
 [Transformer Layers (k+1) ... N] │
          │                     │
          └─── Verification Stage (Full forward pass) ────► Final Output

