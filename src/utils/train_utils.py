import torch


def train(model, device, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    best_val_accuracy = 0.0
    # training and validation loop
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0

        # training
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # validation
        model.eval()
        num_correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                predicted = torch.argmax(y_pred, dim=1)
                loss = criterion(y_pred, y)
                num_correct += (predicted == y).sum()
                total += y.size(0)
                val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))
            val_accuracy = num_correct / total
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), "checkpoints/best_model.pth")

        print(
            f"Epoch {epoch + 1}: train_loss={train_losses[-1]}, val_loss={val_losses[-1]}, val_accuracy={val_accuracy}"
        )
    return train_losses, val_losses, best_val_accuracy
