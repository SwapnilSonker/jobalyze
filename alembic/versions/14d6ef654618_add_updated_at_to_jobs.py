"""add_updated_at_to_jobs

Revision ID: 14d6ef654618
Revises: b050cc94f380
Create Date: 2026-02-05 01:09:56.679699

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '14d6ef654618'
down_revision: Union[str, None] = 'b050cc94f380'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add updated_at column to jobs table
    op.add_column('jobs', sa.Column('updated_at', sa.DateTime(), nullable=True))
    
    # Set existing rows to have updated_at = created_at
    op.execute("UPDATE jobs SET updated_at = created_at WHERE updated_at IS NULL")


def downgrade() -> None:
    # Remove updated_at column
    op.drop_column('jobs', 'updated_at')
